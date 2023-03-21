
import pathlib


class KnowledgeGenerator(object):
    """ class to treat the knowledge generation """

    def __init__(self, model, args):
        super(KnowledgeGenerator, self).__init__()
        self.train_data = model.train_data
        self.data = model.train_data
        self.dataset = args.dataset
        self.clause_stats = []
        self.delete_files()

    @property
    def knowledge(self):
        self.generate_knowledge()
        return f'{self.dataset}_knowledge_base'

    def delete_files(self):
        """
        Deletes knowledge base and datastats file that might
        still be in directory from previous runs
        """
        know_base = pathlib.Path(f'{self.dataset}_knowledge_base')
        if know_base.is_file():
            know_base.unlink()

    def generate_knowledge(self):
        """
        creates the knowledge file based on unary predicates = document classes
        cite is binary predicate
        num_classes int
        """
        assert hasattr(self.data, 'num_classes')

        _clauses = list(range(self.data.num_classes))
        class_list = []
        for i in _clauses:
            class_list += ['class_' + str(i)]

        # Generate knowledge
        kb = ''

        for c in range(self.data.num_classes):
            kb += 'class_' + str(c) + ','

        kb = kb[:-1] + '\nLink\n\n'

        # No unary clauses
        kb = kb[:-1] + '\n>\n'

        # Binary clauses
        # eg: nC(x),nCite(x.y),C(y)
        for c in class_list:
            kb += '_:n' + c + '(x),nLink(x.y),' + c + '(y)\n'

        with open(f'{self.dataset}_knowledge_base', 'w') as kb_file:
            kb_file.write(kb)

