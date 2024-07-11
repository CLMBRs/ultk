from yaml import dump

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from examples.learn_quant.grammar import quantifiers_grammar

if __name__ == "__main__":
    out = quantifiers_grammar.generate()
    print(out)