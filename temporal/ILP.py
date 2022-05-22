from gurobipy import *
from collections import OrderedDict
import numpy as np
import torch


class ILPSolver:

    def __init__(self, pairs, probs, label2idx, flip=False, reverse_relation_map=None):
        """
        :param pairs: list of str tuple ; (docid_eventid, docid_eventid)
        :param probs: a numpy matrix of local prediction scores; (#instance, #classes)
        :param reverse_relation_map: the map from some relation R1 to its reversed one R2
            for example, reverse_relation_map[BEFORE] = AFTER
        """
        if flip:
            # safely check the input pairs
            offset = len(pairs) // 2
            for i in range(len(pairs) // 2):
                assert pairs[i][0] == pairs[i + offset][1] and pairs[i][1] == pairs[i + offset][0]

        self.model = Model('event_relation')

        self.pairs = pairs
        self.idx2pair = {n: self.pairs[n] for n in range(len(pairs))}
        self.pair2idx = {v: k for k, v in self.idx2pair.items()}
        self.probs = probs
        self.label2idx = label2idx
        self.idx2label = OrderedDict([(v, k) for k, v in label2idx.items()])
        self.flip = flip
        self.reverse_relation_map = reverse_relation_map if reverse_relation_map is not None \
            else OrderedDict([('VAGUE', 'VAGUE'),
                              ('BEFORE', 'AFTER'),
                              ('AFTER', 'BEFORE'),
                              ('SIMULTANEOUS', 'SIMULTANEOUS'),
                              ('INCLUDES', 'IS_INCLUDED'),
                              ('IS_INCLUDED', 'INCLUDES')])
        self.num_instances, self.num_classes = probs.shape
        self.pred_labels = list(np.argmax(probs, axis=1))

    def define_vars(self, **kwargs):
        """
        :param kwargs: the kwargs is the placeholder, which is for the additional params [lb, ub, obj, column]
        :return:
        """
        var_table = []
        for n in range(self.num_instances):
            sample = []
            for p in range(self.num_classes):
                sample.append(
                    self.model.addVar(vtype=GRB.BINARY, name="y_%s_%s" % (n, p), **kwargs)
                )
            var_table.append(sample)
        return var_table

    def objective(self, var_table):
        """
        :param var_table: the `var_table` is returned by the previous `define_vars`
        :return:
        """
        obj = 0.0
        for n in range(self.num_instances):
            for p in range(self.num_classes):
                obj += var_table[n][p] * self.probs[n][p]
        return obj

    def single_label(self, sample):
        return sum(sample) == 1

    def transitivity_list(self):
        """
        make triplet
        """
        transitivity_samples = []
        for k, (e1, e2) in self.idx2pair.items():
            for (re1, re2), i in self.pair2idx.items():
                if e2 == re1 and (e1, re2) in self.pair2idx.keys():
                    e3 = re2
                    transitivity_samples.append((self.pair2idx[(e1, e2)],
                                                 self.pair2idx[(e2, e3)],
                                                 self.pair2idx[(e1, e3)]))
        return transitivity_samples

    def transitivity_criteria(self, tab, triplet):
        r1, r2, r3 = triplet
        ld = self.label2idx
        if 'INCLUDES' in ld.keys():
            return [
                tab[r1][ld['BEFORE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']],
                tab[r1][ld['AFTER']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']],
                tab[r1][ld['SIMULTANEOUS']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['SIMULTANEOUS']],
                tab[r1][ld['INCLUDES']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['INCLUDES']],
                tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['IS_INCLUDED']],
                # tab[r1][ld['VAGUE']] + tab[r2][ld['VAGUE']] - tab[r3][ld['VAGUE']],
                tab[r1][ld['BEFORE']] + tab[r2][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][
                    ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['BEFORE']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['INCLUDES']],
                tab[r1][ld['BEFORE']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['AFTER']] + tab[r2][ld['VAGUE']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][
                    ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['AFTER']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][
                    ld['INCLUDES']],
                tab[r1][ld['AFTER']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['INCLUDES']] + tab[r2][ld['VAGUE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['AFTER']] -
                tab[r3][ld['VAGUE']] - tab[r3][ld['BEFORE']],
                tab[r1][ld['INCLUDES']] + tab[r2][ld['BEFORE']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['BEFORE']],
                tab[r1][ld['INCLUDES']] + tab[r2][ld['AFTER']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['AFTER']],
                tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['VAGUE']] - tab[r3][ld['IS_INCLUDED']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['BEFORE']] - tab[r3][ld['AFTER']],
                tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['VAGUE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']] - tab[r3][
                    ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['VAGUE']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']] - tab[r3][
                    ld['INCLUDES']] - tab[r3][ld['IS_INCLUDED']],
                tab[r1][ld['VAGUE']] + tab[r2][ld['INCLUDES']] - tab[r3][ld['INCLUDES']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['BEFORE']] - tab[r3][ld['AFTER']],
                tab[r1][ld['VAGUE']] + tab[r2][ld['IS_INCLUDED']] - tab[r3][ld['IS_INCLUDED']] - tab[r3][ld['VAGUE']] -
                tab[r3][ld['BEFORE']] - tab[r3][ld['AFTER']],
                tab[r1][ld['BEFORE']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['BEFORE']],
                tab[r1][ld['AFTER']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['AFTER']],
                tab[r1][ld['INCLUDES']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['INCLUDES']],
                tab[r1][ld['IS_INCLUDED']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['IS_INCLUDED']],
            ]
        else:
            return [
                tab[r1][ld['BEFORE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']],
                tab[r1][ld['AFTER']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']],
                tab[r1][ld['SIMULTANEOUS']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['SIMULTANEOUS']],
                tab[r1][ld['BEFORE']] + tab[r2][ld['VAGUE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']],
                tab[r1][ld['AFTER']] + tab[r2][ld['VAGUE']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']],
                tab[r1][ld['VAGUE']] + tab[r2][ld['BEFORE']] - tab[r3][ld['BEFORE']] - tab[r3][ld['VAGUE']],
                tab[r1][ld['VAGUE']] + tab[r2][ld['AFTER']] - tab[r3][ld['AFTER']] - tab[r3][ld['VAGUE']],
                tab[r1][ld['BEFORE']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['BEFORE']],
                tab[r1][ld['AFTER']] + tab[r2][ld['SIMULTANEOUS']] - tab[r3][ld['AFTER']],
            ]

    def symmetry_constraints(self, samples, n):
        # Note: Here is assert samples is made up by (normal instance, reverse instance)
        constraints = []
        offset = int(len(self.pairs) / 2)
        for label, idx in self.label2idx.items():
            rev_idx = self.label2idx[self.reverse_relation_map[label]]
            constraints.append(samples[n][idx] == samples[n + offset][rev_idx])
        return constraints

    def define_constraints(self, var_table):
        # Constraint 1: single label assignment (it means that the output of y should be summed to 1)
        for n in range(self.num_instances):
            self.model.addConstr(self.single_label(var_table[n]), "c1_%s" % n)

        # Constraint 2: transitivity
        trans_triples = self.transitivity_list()
        t = 0
        for triple in trans_triples:
            for ci in self.transitivity_criteria(var_table, triple):
                self.model.addConstr(ci <= 1, "c2_%s" % t)
                t += 1

        if self.flip:
            # Constraint 3: Symmetry
            offset = int(len(self.pairs) / 2)
            for n in range(offset):
                for si in self.symmetry_constraints(var_table, n):
                    self.model.addConstr(si, "c3_%s" % n)

    def run(self):
        try:
            # Define variables
            var_table = self.define_vars()

            # Set objective
            self.model.setObjective(self.objective(var_table), GRB.MAXIMIZE)

            # Define constrains
            self.define_constraints(var_table)

            # run model
            self.model.setParam('OutputFlag', False)
            self.model.optimize()

        except GurobiError:
            print('Error reported')

    def inference(self):
        """
        Doing the MAP inference on the given probs
        """
        self.run()

        count = 0
        for v in self.model.getVars():
            # sample idx
            s_idx = int(v.varName.split('_')[1])
            # sample class index
            c_idx = int(v.varName.split('_')[2])

            if v.x == 1.0:
                if self.pred_labels[s_idx] != c_idx:
                    self.pred_labels[s_idx] = c_idx
                    count += 1
        # print('# of global correction: %s' % count)
        # print('Objective Function Value:', self.model.objVal)
        return self.pred_labels


if __name__ == "__main__":
    torch.random.manual_seed(42)
    label2idx = {
        'VAGUE': 0,
        'BEFORE': 1,
        'AFTER': 2,
        'SIMULTANEOUS': 3,
        'INCLUDES': 4,
        'IS_INCLUDED': 5
    }
    pairs = [('exp_1', 'exp_2'),
             ('exp_2', 'exp_3'),
             ('exp_1', 'exp_3'),
             ('exp_4', 'exp_5'),
             ('exp_5', 'exp_6'),
             ('exp_4', 'exp_6'),
             ('exp_7', 'exp_8'),
             # reversed
             ('exp_2', 'exp_1'),
             ('exp_3', 'exp_2'),
             ('exp_3', 'exp_1'),
             ('exp_5', 'exp_4'),
             ('exp_6', 'exp_5'),
             ('exp_6', 'exp_4'),
             ('exp_8', 'exp_7')]
    N_EXP = len(pairs)
    probs = torch.rand(N_EXP, len(label2idx), dtype=torch.float).softmax(-1).numpy()

    solver = ILPSolver(pairs, probs, label2idx, flip=True)
    x = solver.inference()
    print(x)
    # print(x[:probs.shape[0] // 2, :])



