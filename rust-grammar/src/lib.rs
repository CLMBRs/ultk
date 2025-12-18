use ahash::{AHashMap, AHashSet};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Represents a grammar rule with its components
#[derive(Clone, Debug)]
pub struct Rule {
    pub name: String,
    pub lhs: String,
    pub rhs: Option<Vec<String>>,
    pub weight: f64,
}

impl Rule {
    fn is_terminal(&self) -> bool {
        self.rhs.is_none()
    }
}

/// A grammatical expression - the core structure we enumerate
/// We use indices and Arc for efficient cloning and memory usage
#[derive(Clone, Debug)]
pub struct GrammaticalExpression {
    pub rule_name: String,
    pub children: Option<Arc<Vec<GrammaticalExpression>>>,
    pub length: usize,
}

impl GrammaticalExpression {
    fn new(rule_name: String, children: Option<Vec<GrammaticalExpression>>) -> Self {
        let length = 1 + children
            .as_ref()
            .map_or(0, |c| c.iter().map(|e| e.length).sum());
        Self {
            rule_name,
            children: children.map(Arc::new),
            length,
        }
    }

    /// Convert to a string representation matching Python's __str__
    pub fn to_string(&self) -> String {
        match &self.children {
            None => self.rule_name.clone(),
            Some(children) => {
                let child_strs: Vec<String> = children.iter().map(|c| c.to_string()).collect();
                format!("{}({})", self.rule_name, child_strs.join(", "))
            }
        }
    }

    /// Evaluate the expression on a Python universe by calling back to Python
    /// Returns a hashable Python object representing the meaning
    pub fn evaluate_py(
        &self,
        py: Python,
        universe: &PyAny,
        func_cache: &mut FunctionCache,
    ) -> PyResult<PyObject> {
        // Get the function for this rule
        let func = func_cache.get_function(py, &self.rule_name)?;

        match &self.children {
            None => {
                // Terminal: call func with each referent and collect results
                let referents = universe.getattr("referents")?;
                let referents_list: &PyList = referents.downcast()?;

                let results = PyDict::new(py);
                for referent in referents_list.iter() {
                    let result = func.call1(py, (referent,))?;
                    results.set_item(referent, result)?;
                }

                // Return a frozen representation
                Ok(self.freeze_dict(py, results)?)
            }
            Some(children) => {
                // Non-terminal: evaluate children first, then apply func
                let referents = universe.getattr("referents")?;
                let referents_list: &PyList = referents.downcast()?;

                // Evaluate all children for each referent
                let child_results: Vec<PyObject> = children
                    .iter()
                    .map(|c| c.evaluate_py(py, universe, func_cache))
                    .collect::<PyResult<Vec<_>>>()?;

                let results = PyDict::new(py);
                for referent in referents_list.iter() {
                    // Get child values for this referent
                    let child_values: Vec<PyObject> = child_results
                        .iter()
                        .map(|cr| cr.as_ref(py).get_item(referent).map(|v| v.to_object(py)))
                        .collect::<PyResult<Vec<_>>>()?;

                    // Call the function with child values
                    let args = PyTuple::new(py, &child_values);
                    let result = func.call1(py, args)?;
                    results.set_item(referent, result)?;
                }

                Ok(self.freeze_dict(py, results)?)
            }
        }
    }

    /// Convert dict to a hashable frozen representation
    fn freeze_dict(&self, py: Python, dict: &PyDict) -> PyResult<PyObject> {
        // Convert to a frozen tuple of (key, value) pairs for hashing
        let items: Vec<(PyObject, PyObject)> = dict
            .items()
            .iter()
            .map(|item| {
                let tuple: &PyTuple = item.downcast().unwrap();
                (
                    tuple.get_item(0).unwrap().to_object(py),
                    tuple.get_item(1).unwrap().to_object(py),
                )
            })
            .collect();

        let frozen_tuple = PyTuple::new(py, items);
        Ok(frozen_tuple.to_object(py))
    }
}

/// Cache for Python function objects to avoid repeated lookups
pub struct FunctionCache {
    functions: HashMap<String, PyObject>,
}

impl FunctionCache {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    fn add_function(&mut self, name: String, func: PyObject) {
        self.functions.insert(name, func);
    }

    fn get_function<'py>(&self, py: Python<'py>, name: &str) -> PyResult<&PyObject> {
        self.functions.get(name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Function not found: {}", name))
        })
    }
}

/// The main Grammar struct with enumeration logic
#[pyclass]
pub struct RustGrammar {
    start: String,
    rules: HashMap<String, Vec<Rule>>,
    rules_by_name: HashMap<String, Rule>,
    func_cache: FunctionCache,
}

#[pymethods]
impl RustGrammar {
    #[new]
    fn new(start: String) -> Self {
        Self {
            start,
            rules: HashMap::new(),
            rules_by_name: HashMap::new(),
            func_cache: FunctionCache::new(),
        }
    }

    /// Add a rule to the grammar
    fn add_rule(
        &mut self,
        name: String,
        lhs: String,
        rhs: Option<Vec<String>>,
        weight: f64,
        func: PyObject,
    ) {
        let rule = Rule {
            name: name.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            weight,
        };

        self.rules
            .entry(lhs)
            .or_insert_with(Vec::new)
            .push(rule.clone());
        self.rules_by_name.insert(name.clone(), rule);
        self.func_cache.add_function(name, func);
    }

    /// Main entry point: get unique expressions up to a given depth
    /// This is the performance-critical function we're optimizing
    fn get_unique_expressions(
        &mut self,
        py: Python,
        depth: usize,
        universe: &PyAny,
        unique_key_fn: PyObject,
        compare_fn: PyObject,
        lhs: Option<String>,
        max_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let start_lhs = lhs.unwrap_or_else(|| self.start.clone());
        let max = max_size.unwrap_or(usize::MAX);

        // Dictionary to store unique expressions: lhs -> (key -> expression)
        let mut unique_dict: AHashMap<String, AHashMap<u64, GrammaticalExpression>> =
            AHashMap::new();
        unique_dict.insert(start_lhs.clone(), AHashMap::new());

        // Cache for enumeration results
        let mut cache: AHashMap<(usize, String), Vec<GrammaticalExpression>> = AHashMap::new();

        // Enumerate expressions at each depth
        for d in 0..depth {
            let expressions = self.enumerate_at_depth(py, d, &start_lhs, &mut cache)?;

            // Process each expression for uniqueness
            for expr in expressions {
                if unique_dict[&start_lhs].len() >= max {
                    break;
                }

                // Evaluate expression to get its key
                let meaning = expr.evaluate_py(py, universe, &mut self.func_cache)?;
                let key_obj = unique_key_fn.call1(py, (meaning,))?;

                // Hash the key for efficient lookups
                let key_hash = self.hash_pyobject(py, &key_obj)?;

                // Check if we should add this expression
                let lhs_dict = unique_dict.get_mut(&start_lhs).unwrap();
                let should_add = if let Some(existing) = lhs_dict.get(&key_hash) {
                    // Compare using the provided function
                    let result = compare_fn.call1(py, (expr.length, existing.length))?;
                    result.extract::<bool>(py)?
                } else {
                    true
                };

                if should_add {
                    lhs_dict.insert(key_hash, expr);
                }
            }

            if unique_dict[&start_lhs].len() >= max {
                break;
            }
        }

        // Convert to Python dict
        self.to_python_dict(py, &unique_dict[&start_lhs])
    }

    /// Hash a Python object for use as a dictionary key
    fn hash_pyobject(&self, py: Python, obj: &PyObject) -> PyResult<u64> {
        let hash_val = obj.as_ref(py).hash()?;
        Ok(hash_val as u64)
    }

    /// Convert our internal representation to a Python dict
    fn to_python_dict(
        &self,
        py: Python,
        exprs: &AHashMap<u64, GrammaticalExpression>,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (i, expr) in exprs.values().enumerate() {
            dict.set_item(i, expr.to_string())?;
        }
        Ok(dict.to_object(py))
    }
}

impl RustGrammar {
    /// Enumerate all expressions at a specific depth
    /// This is the core recursive function that's called many times
    fn enumerate_at_depth(
        &mut self,
        py: Python,
        depth: usize,
        lhs: &str,
        cache: &mut AHashMap<(usize, String), Vec<GrammaticalExpression>>,
    ) -> PyResult<Vec<GrammaticalExpression>> {
        let cache_key = (depth, lhs.to_string());

        // Return from cache if available
        if let Some(cached) = cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut results = Vec::new();

        if depth == 0 {
            // Base case: only terminal rules
            if let Some(rules) = self.rules.get(lhs) {
                for rule in rules {
                    if rule.is_terminal() {
                        results.push(GrammaticalExpression::new(rule.name.clone(), None));
                    }
                }
            }
        } else {
            // Recursive case: expand non-terminal rules
            if let Some(rules) = self.rules.get(lhs) {
                for rule in rules {
                    if let Some(rhs) = &rule.rhs {
                        // Generate all combinations of child depths that sum to depth-1
                        let num_children = rhs.len();
                        let combinations = generate_depth_combinations(depth, num_children);

                        for child_depths in combinations {
                            // Only consider if at least one child is at max depth
                            if child_depths.iter().max().unwrap() < &(depth - 1) {
                                continue;
                            }

                            // Get expressions for each child
                            let mut child_expr_lists = Vec::new();
                            for (child_depth, child_lhs) in child_depths.iter().zip(rhs.iter()) {
                                let child_exprs =
                                    self.enumerate_at_depth(py, *child_depth, child_lhs, cache)?;
                                child_expr_lists.push(child_exprs);
                            }

                            // Generate all combinations of children
                            let child_combinations = cartesian_product(&child_expr_lists);

                            for children in child_combinations {
                                results.push(GrammaticalExpression::new(
                                    rule.name.clone(),
                                    Some(children),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Cache the results
        cache.insert(cache_key, results.clone());
        Ok(results)
    }
}

/// Generate all ways to distribute depth among num_slots children
fn generate_depth_combinations(depth: usize, num_slots: usize) -> Vec<Vec<usize>> {
    if num_slots == 0 {
        return vec![vec![]];
    }
    if num_slots == 1 {
        return vec![vec![depth]];
    }

    let mut results = Vec::new();
    for first_depth in 0..depth {
        let rest = generate_depth_combinations(depth - first_depth, num_slots - 1);
        for mut r in rest {
            let mut combo = vec![first_depth];
            combo.append(&mut r);
            results.push(combo);
        }
    }
    results
}

/// Compute Cartesian product of vectors efficiently
fn cartesian_product(lists: &[Vec<GrammaticalExpression>]) -> Vec<Vec<GrammaticalExpression>> {
    if lists.is_empty() {
        return vec![vec![]];
    }

    let mut results = vec![vec![]];

    for list in lists {
        let mut new_results = Vec::new();
        for existing in &results {
            for item in list {
                let mut new_combo = existing.clone();
                new_combo.push(item.clone());
                new_results.push(new_combo);
            }
        }
        results = new_results;
    }

    results
}

/// Python module definition
#[pymodule]
fn _ultk_grammar(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustGrammar>()?;
    Ok(())
}
