"""generates the knowledge of type CLASS(x) AND CITE(X.Y) => CLASS(Y) for each class """


def generate_knowledge(num_classes):
    """
    creates the knowledge file based on unary predicates = document classes
    cite is binary predicate
    num_classes int
    """

    class_list = list(range(num_classes))
    #class_list = []
    for i in class_list:
        class_list[i] = 'class_' + str(i)
    # Generate knowledge
    kb = ''

    # List of predicates
    for c in class_list:
        kb += c + ','


    kb = kb[:-1] + '\nCite\n\n'

    # No unary clauses

    kb = kb[:-1] + '\n>\n'

    # Binary clauses

    # nC(x),nCite(x.y),C(y)
    for c in class_list[:3]:
        kb += '_:n' + c + '(x),nCite(x.y),' + c + '(y)\n'

    with open('knowledge_base', 'w') as kb_file:
        kb_file.write(kb)

    return kb

if __name__ == '__main__':
    kb = generate_knowledge(40)
