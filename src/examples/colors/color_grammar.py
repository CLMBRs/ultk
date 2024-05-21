import numpy as np
from ultk.language.language import Language, Expression

class ColorLanguage(Language):
    """Language representing a set of color probabilities. 

    Args:
        Language (_type_): _description_
    """
    def __init__(self, expressions: tuple[Expression, ...], natural:bool = True, **kwargs):
        super().__init__(expressions, **kwargs)
        self.natural = natural

    natural:bool = True
    def is_natural(self) -> bool:
        return self.natural
    
    def __getstate__(self):
       """Used for serializing instances"""
       
       # start with a copy so we don't accidentally modify the object state
       # or cause other conflicts
       state = self.__dict__.copy()

       # Print unpicklable entries
       if('f' in state):
            # print(f"Unpickled: {state['f']}")
            # remove unpicklable entries
            del state['f']

       
    
    def __setstate__(self, state):
        """Used for deserializing instances"""
        # restore instance attributes
        self.__dict__.update(state)
    
    def centroid(self) -> np.ndarray:
        """Generate the centroid of the referents in the language.

        Args:
            None
        Returns:
            dict[expression form, centroid]: map of expressions to their corresponding centroids in the CIELab color space.
        """
        major_term_map = {}
        for major_term in self.expressions:
            centroid = np.zeros(3)
            
            for referent in major_term.meaning.referents:
                centroid += np.array((referent.L, referent.a, referent.b)) * major_term.meaning._dist[referent.name]
            centroid /= len(major_term.meaning.referents)
            major_term.centroid = centroid

            major_term_map[major_term.form] = centroid
        return major_term_map