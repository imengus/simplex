import numpy as np
import re


class Tableau:
    """Generate and operate on simplex tableaus"""

    def __init__(self, lin_prog, n_art_vars):
        """Initialise Tableau class

        Args:
            lin_prog : list[str]
                Line separated string input in list
            n_art_vars : int
                Number of artificial/ surplus variables needed
        """
        self.n_art_vars = n_art_vars

        # 2 if there are >= or == constraints (nonstandard), 1 otherwise (std)
        self.n_stages = (n_art_vars > 0) + 1

        # Number of decision variables x1, x2, x3...
        self.n_vars = len(lin_prog[0].split()) - 1

        # Number of rows the initial simplex tableau will have
        self.n_rows = len(lin_prog) + (n_art_vars > 0)

        # Number of columns of the initial simplex tableau
        self.n_cols = self.n_vars + self.n_rows + self.n_art_vars + 1

        # Number of slack variables added to make inequalities into equalities
        self.n_slack = len(lin_prog) - 1

        # Initial tableau with no entries
        self.tableau = np.zeros((self.n_rows - (n_art_vars > 0), self.n_cols))

        # Values of non-basic variables following iterations
        self.output_dict = {}

        # Objectives for each stage
        self.objectives = ["max"]

        # Main objective
        self.rhs_sign = 1

        # Indices of rows that have greater than or equal to constraints
        self.geq_ids = []

        self.col_titles = []

        # Index of current pivot row and column
        self.row_idx = None
        self.col_idx = None

        # Does objective row only contain (non)-negative values?
        self.stop_iter = False

    @staticmethod
    def find_type(row_string):
        """Find whether the constraint is a minimise or maximise objective
        function or a constraint with an (in)equality.

        args:
            row_string : str
                Contains one of the input rows

        returns:
            match.group(0) : str
                Type of objective/inequality in min/max/<=/>=/==
        """
        row_string = row_string.lower()
        match = re.search("(min|max|<=|>=|==)", row_string)
        return match.group(0)

    @staticmethod
    def preprocess(lin_prog):
        """Replace equalities with >=, <= and count the number of artificial
        variables necessary.

        args:
            lin_prog : list[str]
                User input

        returns:
            std_lin_prog : list[str]
                User input without "==" constraints
            n_art_vars : int
                Number of artificial variables needed. If greater than 1, two
                stage simplex method is required.
        """
        std_lin_prog = []
        n_art_vars = 0
        for row in lin_prog:
            row_type = Tableau.find_type(row)
            split_row = row.split()

            # These types don't need intervention
            if row_type in "min-max-<=":
                std_lin_prog.append(row)
                continue
            if row_type == "==":

                # == constraints are split into => and <= inequalities, so one
                # artificial variable is introduced per == constraint.
                n_art_vars += 1

                # "==" replaced with "<=" then the row string is reformed
                split_row[-2] = "<="
                std_lin_prog.append(" ".join(split_row))
                split_row[-2] = ">="
                std_lin_prog.append(" ".join(split_row))
            if row_type == ">=":
                n_art_vars += 1
                std_lin_prog.append(row)
        return std_lin_prog, n_art_vars

    def add_row(self, row_string, row_num, art_vars_left=0):
        """Add objective, slack, and artificial variables to row given by
        `row_num` to the simplex tableau

        args:
            row_string : str
                One of the input rows
            row_num : int
                Index of the row, the top being 0th
            art_vars_left : int
                Initially equal to `n_art_vars`. For keeping track of the
                rows in which artificial variables are placed
        """
        row_type = self.find_type(row_string)

        # Initialise row
        row = np.zeros(self.n_cols)

        row_string = row_string.split()

        # Multiplicand of variable entries = Value of slack entry =  1
        coeff = slack_val = 1

        # Not including inequality sign or RHS
        s = slice(0, -2)

        if row_type in "max-min":

            # Change the slice so that it includes the last two positions
            s = slice(1, self.n_cols)

            # No slack variables are added for the objective rows
            slack_val = 0

            self.rhs_sign -= 2 * (row_type == "min")

        else:

            # If it is a constraint row add RHS
            row[-1] = row_string[-1]

        if row_type == "max":

            # Rearranges objective function by moving variables to LHS, hence
            # the change of sign. e.g. "P = x + y" --> "P - x - y = 0"
            coeff = -1
            pass

        if row_type == ">=":
            self.geq_ids.append(row_num)

            # Slack is taken away from inequality to make it an equality
            slack_val = -1

            # Adds artificial variable
            row[-art_vars_left - 1] = 1

            # Move position of next artificial variable one column to the right
            art_vars_left -= 1

        # Add decision variable entries
        row[: self.n_vars] = np.array([coeff * int(x) for x in row_string[s]])

        # Add slack entry
        row[self.n_vars + row_num] = slack_val

        # Add row to tableau
        self.tableau[row_num] = row
        return art_vars_left

    def create_art_row(self):
        """
        Artificial variables created must be minimised. Total of artificial
        variables obtained by summing the rows with >= inequalities. This
        produces the new objective row.
        """
        # Column index of left-most artificial column
        art_idx = self.n_cols - self.n_art_vars - 1

        # Create index array to exclude artificial variable columns in sum
        idx_arr = np.ones(self.n_cols, dtype=bool)
        idx_arr[art_idx:-1] = False

        # Sum rows whose indices belong to geq_ids, i.e. sum >= rows
        art_row = np.sum(self.tableau[:, idx_arr][self.geq_ids], axis=0)

        # Insert 0s to fill the gap between non-artificial variables and RHS
        art_row = np.insert(art_row, art_idx, np.zeros(self.n_art_vars))

        # Add to tableau
        self.tableau = np.vstack((art_row, self.tableau))

        # First objective becomes Min the sum of the artificial variables
        self.objectives.append("min")

    def create_tableau(self, constraints):
        """Adds rows one at a time. If two stage, then the artificial objective
        row is also added.
        """
        art_vars_left = self.n_art_vars
        for row_idx, row_string in enumerate(constraints):
            art_vars_left = self.add_row(row_string, row_idx, art_vars_left)

        # If there are any >= constraints, an additional objective is needed
        if self.geq_ids:
            self.create_art_row()

    def delete_empty(self):
        """Delete empty columns"""
        del_ids = ~np.any(self.tableau, axis=0)
        self.tableau = np.delete(self.tableau, del_ids, axis=1)

    def generate_col_titles(self):
        """Simplex tableau contains variable, slack, artificial, and RHS
        columns e.g. x_1, x_2, s_1, s_2, a_1, RHS
        """
        string_starts = ["x_", "s_", "a_"]
        constants = self.n_vars, self.n_slack, self.n_art_vars
        titles = []
        for i in range(3):
            for j in range(constants[i]):
                titles.append(string_starts[i] + str(j + 1))
        titles.append("RHS")
        self.col_titles = titles

    def find_pivot(self):
        """Finds the pivot row and column."""
        tableau = self.tableau
        objective = self.objectives[-1]

        # Find entries of highest magnitude in objective rows
        sign = (objective == "min") - (objective == "max")
        self.col_idx = np.argmax(sign * tableau[0, :-1])

        # Check if choice is valid, or if iteration must be stopped
        if sign * self.tableau[0, self.col_idx] <= 0:
            self.stop_iter = True
            return

        # Pivot row is chosen as having the lowest quotient when elements of
        # the pivot column divide the right-hand side

        # Slice excluding the objective rows
        s = slice(self.n_stages, self.n_rows)

        # RHS
        dividend = tableau[s, -1]

        # Elements of pivot column within slice
        divisor = tableau[s, self.col_idx]

        # Array filled with nans
        nans = np.full(self.n_rows - self.n_stages, np.nan)

        # If element in pivot column is greater than zero, return quotient
        # or nan otherwise
        quotients = np.divide(dividend, divisor, out=nans, where=divisor > 0)

        # Arg of minimum quotient excluding the nan values. `n_stages` is added
        # to compensate for earlier exclusion of objective columns
        self.row_idx = np.nanargmin(quotients) + self.n_stages

    def pivot(self):
        """Pivots on value on the intersection of pivot row and column."""

        # Avoid changes to original tableau
        piv_row = self.tableau[self.row_idx].copy()

        piv_val = piv_row[self.col_idx]

        # Entry becomes 1
        piv_row *= 1 / piv_val

        # Variable in pivot column becomes basic, ie the only non-zero entry
        for idx, coeff in enumerate(self.tableau[:, self.col_idx]):
            self.tableau[idx] += -coeff * piv_row
        self.tableau[self.row_idx] = piv_row

    def change_stage(self):
        """Exits first phase of the two-stage method by deleting artificial
        rows and columns, or completes the algorithm if exiting the standard
        case."""
        # Objective of original objective row remains
        self.objectives.pop()

        if not self.objectives:
            return

        # Slice containing ids for artificial columns
        s = slice(-self.n_art_vars - 1, -1)

        # Delete the artificial variable columns
        self.tableau = np.delete(self.tableau, s, axis=1)

        # Delete the objective row of the first stage
        self.tableau = np.delete(self.tableau, 0, axis=0)

        self.n_stages = 1
        self.n_rows -= 1
        self.n_art_vars = 0
        self.stop_iter = False

    def run_simp(self):
        """Recursively operate on tableau until objective function cannot be
        improved further"""
        # If optimal solution reached
        if not self.objectives:
            self.interpret_tableau()
            raise Exception

        self.find_pivot()
        if self.stop_iter:
            self.change_stage()
            self.run_simp()
        self.pivot()
        self.run_simp()

    def interpret_tableau(self):
        """Given the final tableau, add the corresponding values of the basic
        variables to the `output_dict`"""

        # P = RHS of final tableau
        self.output_dict["P"] = self.rhs_sign * self.tableau[0, -1]

        n_current_cols = len(self.tableau[0])
        for i in range(n_current_cols):

            # Gives ids of nonzero entries in the ith column
            nonzero = np.nonzero(self.tableau[:, i])
            n_nonzero = len(nonzero[0])

            # First entry in the nonzero ids
            nonzero_rowidx = nonzero[0][0]
            nonzero_val = self.tableau[nonzero_rowidx, i]

            # If there is only one nonzero value in column, which is one
            if n_nonzero == nonzero_val == 1:
                rhs_val = self.tableau[nonzero_rowidx, -1]
                self.output_dict[self.col_titles[i]] = rhs_val
