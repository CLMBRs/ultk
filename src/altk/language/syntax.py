"""Classes for modeling the forms (and structures) of a language.

    Minimally, a form is a way of distinguishing lexical items in a language, which can be thought of a packages of form and meaning. Meanings do not need to be unique, and neither do forms (consider e.g., synoymy and homophony) but you should think about how to distinguish two expressions in principle.

    In efficient communication analyses, it is possible that the simplicity/informativeness trade-off is affected by the complexity of grammatical forms.

    Typical usage example:

        from altk.language.syntax import Form
        from altk.language.language import Expression, Language
        form = Form('blue')
        meaning = Color_Meaning() # some default meaning
        expression = Expression(form, meaning)
        lang = Language([expression])

"""
