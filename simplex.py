import numpy as np


class Tableau:
    """Operate on simplex tableaus"""

    def __init__(self, tableau: np.ndarray, n_vars: int) -> None:
        if tableau is None:
            return

        self.tableau = tableau
        self.n_rows = len(tableau[:, 0])
        self.n_cols = len(tableau[0])

        # Number of decision variables x1, x2, x3...
        self.n_vars = n_vars

        self.n_art_vars = len(np.where(tableau[self.n_vars : -1] == -1)[0])

        # 2 if there are >= or == constraints (nonstandard), 1 otherwise (std)
        self.n_stages = (self.n_art_vars > 0) + 1
        # Number of slack variables added to make inequalities into equalities
        self.n_slack = self.n_rows - self.n_stages

        # Objectives for each stage
        self.objectives = ["max"]

        if self.n_art_vars:
            self.objectives.append("min")

        self.col_titles = None

        # Index of current pivot row and column
        self.row_idx = None
        self.col_idx = None

        # Does objective row only contain (non)-negative values?
        self.stop_iter = False

    def generate_col_titles(self, n_vars=None, n_slack=None, n_art_vars=None):
        """Generate column titles for tableau of specific dimensions

        >>> Tableau.generate_col_titles(None, 2, 3, 1)
        ['x1', 'x2', 's1', 's2', 's3', 'a1', 'RHS']
        """
        constants = n_vars, n_slack, n_art_vars
        if not all(constants):
            constants = self.n_vars, self.n_slack, self.n_art_vars
        string_starts = ["x", "s", "a"]
        titles = []
        for i in range(3):
            for j in range(constants[i]):
                titles.append(string_starts[i] + str(j + 1))
        titles.append("RHS")
        return titles

    def find_pivot(self, tableau):
        """Finds the pivot row and column.
        >>> tableau = np.array([[-2,-1,0,0,0],[3,1,1,0,6],[1,2,0,1,7.]])
        >>> t = Tableau(tableau, 2)
        >>> t.find_pivot(t.tableau)
        (1, 0)
        """
        objective = self.objectives[-1]

        # Find entries of highest magnitude in objective rows
        sign = (objective == "min") - (objective == "max")
        col_idx = np.argmax(sign * tableau[0, : self.n_vars])

        # Check if choice is valid, or if iteration must be stopped
        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        # Pivot row is chosen as having the lowest quotient when elements of
        # the pivot column divide the right-hand side

        # Slice excluding the objective rows
        s = slice(self.n_stages, self.n_rows)

        # RHS
        dividend = tableau[s, -1]

        # Elements of pivot column within slice
        divisor = tableau[s, col_idx]

        # Array filled with nans
        nans = np.full(self.n_rows - self.n_stages, np.nan)

        # If element in pivot column is greater than zeron_stages, return quotient
        # or nan otherwise
        quotients = np.divide(dividend, divisor, out=nans, where=divisor > 0)

        # Arg of minimum quotient excluding the nan values. `n_stages` is added
        # to compensate for earlier exclusion of objective columns
        row_idx = np.nanargmin(quotients) + self.n_stages
        return row_idx, col_idx

    def pivot(self, tableau, row_idx, col_idx):
        """Pivots on value on the intersection of pivot row and column.

        >>> tableau = np.array([[-2,-3,0,0,0],[1,3,1,0,4],[3,1,0,1,4.]])
        >>> t = Tableau(tableau, 2)
        >>> t.pivot(t.tableau, 1, 0).tolist()
        ... # doctest: +NORMALIZE_WHITESPACE
        [[0.0, 3.0, 2.0, 0.0, 8.0],
        [1.0, 3.0, 1.0, 0.0, 4.0],
        [0.0, -8.0, -3.0, 1.0, -8.0]]
        """
        # Avoid changes to original tableau
        piv_row = tableau[row_idx].copy()

        piv_val = piv_row[col_idx]

        # Entry becomes 1
        piv_row *= 1 / piv_val

        # Variable in pivot column becomes basic, ie the only non-zero entry
        for idx, coeff in enumerate(tableau[:, col_idx]):
            tableau[idx] += -coeff * piv_row
        tableau[row_idx] = piv_row
        return tableau

    def change_stage(self, tableau, objectives):
        """Exits first phase of the two-stage method by deleting artificial
        rows and columns, or completes the algorithm if exiting the standard
        case.
        
        >>> tableau = np.array([[3,3,-1,-1,0,0,4],[2,1,0,0,0,0,0.], \
        [1,2,-1,0,1,0,2],[2,1,0,-1,0,1,2]])
        >>> t = Tableau(tableau, 2)
        >>> t.change_stage(t.tableau, t.objectives).tolist()
        ... # doctest: +NORMALIZE_WHITESPACE
        [[2.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
        [1.0, 2.0, -1.0, 0.0, 1.0, 2.0], 
        [2.0, 1.0, 0.0, -1.0, 0.0, 2.0]]
        """
        # Objective of original objective row remains
        objectives.pop()

        if not objectives:
            return tableau

        # Slice containing ids for artificial columns
        s = slice(-self.n_art_vars - 1, -1)

        # Delete the artificial variable columns
        tableau = np.delete(tableau, s, axis=1)

        # Delete the objective row of the first stage
        tableau = np.delete(tableau, 0, axis=0)

        self.n_stages = 1
        self.n_rows -= 1
        self.n_art_vars = 0
        self.stop_iter = False
        self.objectives = objectives
        return tableau

    def run_simplex(self):
        """Operate on tableau until objective function cannot be
        improved further

        # Standard linear program:
        Max:  x1 +  x2
        ST:   x1 + 3x2 <= 4
             3x1 +  x3 <= 4 
        >>> tableau = np.array([[-1,-1,0,0,0],[1,3,1,0,4],[3,1,0,1,4.]])
        >>> t = Tableau(tableau, 2)
        >>> t.run_simplex()
        {'P': 2.0, 'x1': 1.0, 'x2': 1.0}

        # Optimal tableau:
        >>> tableau = np.array([[0,0,0.25,0.25,2],[0,1,0.375,-0.125,1], \
            [1,0,-0.125,0.375,1]])
        >>> t = Tableau(tableau, 2)
        >>> t.run_simplex()
        {'P': 2.0, 'x1': 1.0, 'x2': 1.0}

        # Non standard linear program (>= constraints)
        Max: 2x1 + 3x2 +  x3
        ST:   x1 +  x2 +  x3 <= 40
             2x1 +  x2 -  x3 >= 10
                 -  x2 +  x3 >= 10
        >>> tableau = np.array([[2,0,0,0,-1,-1,0,0,20], \
                                [-2,-3,-1,0,0,0,0,0,0], \
                                [1,1,1,1,0,0,0,0,40], \
                                [2,1,-1,0,-1,0,1,0,10], \
                                [0,-1,1,0,0,-1,0,1,10.]])
        >>> t = Tableau(tableau, 3)
        >>> t.run_simplex()
        {'P': 70.0, 'x1': 10.0, 'x2': 10.0, 'x3': 20.0}

        # Non standard: minimisation and equalities
        Min: x1 +  x2
        ST: 2x1 +  x2 = 12
            6x1 + 5x2 = 40
        >>> tableau = np.array([[8,6,0,-1,0,-1,0,0,52], \
                                [1,1,0,0,0,0,0,0,0], \
                                [2,1,1,0,0,0,0,0,12], \
                                [2,1,0,-1,0,0,1,0,12],\
                                [6,5,0,0,1,0,0,0,40], \
                                [6,5,0,0,0,-1,0,1,40.]])
        >>> t = Tableau(tableau, 2)
        >>> t.run_simplex()
        {'P': -7.0, 'x1': 5.0, 'x2': 2.0}
        """
        iter_num = 0
        while iter_num < 100:
            # Completion of each stage removes an objective. If both stages
            # are complete, then no objectives are left
            if not self.objectives:

                self.col_titles = self.generate_col_titles()

                # Find the values of each variable at optimal solution
                return self.interpret_tableau(self.tableau, self.col_titles)

            row_idx, col_idx = self.find_pivot(self.tableau)
            # If there are no more negative values in objective row
            if self.stop_iter:

                # Delete artifical variable columns and rows. Update attributes.
                self.tableau = self.change_stage(self.tableau, self.objectives)

                # Pivot again
                continue

            self.tableau = self.pivot(self.tableau, row_idx, col_idx)
            iter_num += 1
        return "Maximum iteration reached"

    def interpret_tableau(self, tableau, col_titles):
        """Given the final tableau, add the corresponding values of the basic
        decision variables to the `output_dict`
        >>> tableau = np.array([[0,0,0.875,0.375,5],[0,1,0.375,-0.125,1], \
            [1,0,-0.125,0.375,1]])
        >>> t = Tableau(tableau, 2)
        >>> t.interpret_tableau(tableau, ["x1", "x2", "s1", "s2", "RHS"])
        {'P': 5.0, 'x1': 1.0, 'x2': 1.0}
        """
        output_dict = {}

        # P = RHS of final tableau
        output_dict["P"] = tableau[0, -1]

        for i in range(self.n_vars):

            # Gives ids of nonzero entries in the ith column
            nonzero = np.nonzero(tableau[:, i])
            n_nonzero = len(nonzero[0])

            # First entry in the nonzero ids
            nonzero_rowidx = nonzero[0][0]
            nonzero_val = tableau[nonzero_rowidx, i]

            # If there is only one nonzero value in column, which is one
            if n_nonzero == nonzero_val == 1:
                rhs_val = tableau[nonzero_rowidx, -1]
                output_dict[col_titles[i]] = rhs_val
        for title in col_titles:
            if title[0] not in "R-s":
                output_dict.setdefault(title, 0)
        return output_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()
