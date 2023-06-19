import re
import numpy as np


class Tableau:
    """Operate on simplex tableaus"""

    def __init__(self, pre_lin_prog=None, tableau=None, n_vars=None):
        # If string input
        if pre_lin_prog is not None:
            lin_prog, self.n_art_vars = self.preprocess(pre_lin_prog)

            # 2 if there are >= or == constraints (nonstandard), 1 otherwise
            self.n_stages = (self.n_art_vars > 0) + 1

            # Number of decision variables x1, x2, x3...
            self.n_vars = len(lin_prog[0].split()) - 1

            # Number of rows the initial simplex tableau will have
            self.n_rows = len(lin_prog) + (self.n_art_vars > 0)

            # Number of columns of the initial simplex tableau
            self.n_cols = self.n_vars + self.n_rows + self.n_art_vars + 1

            # Number of slack vars added to make inequalities into equalities
            self.n_slack = self.n_rows - self.n_stages

            # Indices of rows that have greater than or equal to constraints
            self.geq_ids = []

            self.tableau = self.create_tableau(lin_prog)

        # If simplex tableau input
        if tableau is not None:
            # Check if RHS is negative
            if np.any(tableau[:, -1], where=tableau[:, -1] < 0):
                raise ValueError("RHS must be > 0")

            self.tableau = tableau
            self.n_rows, _ = tableau.shape

            # Number of decision variables x1, x2, x3...
            self.n_vars = n_vars

            # Number of artificial variables to be minimised
            self.n_art_vars = len(np.where(tableau[self.n_vars : -1] == -1)[0])

            # 2 if there are >= or == constraints (nonstandard), 1 otherwise
            self.n_stages = (self.n_art_vars > 0) + 1

            # Number of slack vars added to make inequalities into equalities
            self.n_slack = self.n_rows - self.n_stages

        # Objectives for each stage
        self.objectives = ["max"]

        # In two stage simplex, first minimise then maximise
        if self.n_art_vars:
            self.objectives.append("min")

        self.col_titles = self.generate_col_titles(
            self.n_vars, self.n_slack, self.n_art_vars
        )

        # Index of current pivot row and column
        self.row_idx = None
        self.col_idx = None

        # Does objective row only contain (non)-negative values?
        self.stop_iter = False

        self.output_dict = {}

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

    @staticmethod
    def find_type(row_string):
        """Find whether the constraint is a minimise or maximise objective 
        function or a constraint with an (in)equality

        args:
            row_string : str
                Contains one of the input rows

        returns:
            _ : str
                Type of objective/inequality in min/max/<=/>=/==
        """
        row_string = row_string.lower()
        match = re.search("(min|max|<=|>=|==)", row_string)
        return match.group(0)

    def add_row(self, row_string, row_idx, art_left=0):
        """Add objective, slack, and artificial variables to row given by 
        `row_num` to the simplex tableau

        args:
            row_string : str
                One of the input rows
            row_idx : int
                Index of the row, the top being 0th
            art_left : int
                Initially equal to `n_art_vars`. For keeping track of the
                rows in which artificial variables are placed

        returns:
            row : np.ndarray
                New row with decision, slack, artificial and RHS variables
            art_left : int
                1 less than initial value if artificial variable added to row

        """
        row_type = self.find_type(row_string)

        row_string = row_string.split()

        # Initialise row
        row = np.zeros(self.n_cols)

        # Multiplicand of variable entries = Value of slack entry =  1
        coeff = slack_val = 1

        # Not including inequality sign or RHS
        s = slice(0, -2)

        if row_type in "max-min":

            # Change the slice so that it includes the last two positions
            s = slice(1, self.n_cols)

            # No slack variables are added for the objective rows
            slack_val = 0

        else:

            # If it is a constraint row add RHS
            row[-1] = row_string[-1]

        if row_type == "max":

            # Rearranges objective function by moving variables to LHS, hence
            # the change of sign. e.g. "P = x + y" --> "P - x - y = 0"
            coeff = -1
            pass

        if row_type == ">=":
            self.geq_ids.append(row_idx)

            # Slack is taken away from inequality to make it an equality
            slack_val = -1

            # Adds artificial variable
            row[-art_left - 1] = 1

            # Move position of next artificial variable one column to the right
            art_left -= 1

        # Add decision variable entries
        row[: self.n_vars] = np.array([coeff * int(x) for x in row_string[s]])

        # Add slack entry
        row[self.n_vars + row_idx] = slack_val
        return row, art_left

    def create_art_row(self, tableau):
        """
        Artificial variables created must be minimised. Total of artificial
        variables obtained by summing the rows with >= inequalities. This
        produces the new objective row.

        args:
            tableau : np.ndarray
                simplex tableau

        returns:
            art_row : np.ndarray
                new objective row
        """
        # Column index of left-most artificial column
        art_idx = self.n_cols - self.n_art_vars - 1

        # Create index array to exclude artificial variable columns in sum
        idx_arr = np.ones(self.n_cols, dtype=bool)
        idx_arr[art_idx:-1] = False

        # Sum rows whose indices belong to geq_ids, i.e. sum >= rows
        art_row = np.sum(tableau[:, idx_arr][self.geq_ids], axis=0)

        # Insert 0s to fill the gap between non-artificial variables and RHS
        art_row = np.insert(art_row, art_idx, np.zeros(self.n_art_vars))

        return art_row

    @staticmethod
    def delete_empty(tableau):
        """Delete empty columns

        args:
            tableau : np.ndarray
                simplex tableau

        returns:
                _ : np.ndarray
                tableau without empty columns
        """
        del_ids = ~np.any(tableau, axis=0)
        return np.delete(tableau, del_ids, axis=1)

    def create_tableau(self, lin_prog):
        """Adds rows one at a time. If two stage, then the artificial objective
        row is also added.

        args:
            lin_prog : list[str]
                list with each element being a row of the linear program

        returns:
            _ : np.ndarray
                final simplex tableau
        """
        tableau = np.zeros((self.n_rows - (self.n_art_vars > 0), self.n_cols))
        art_left = self.n_art_vars
        for row_idx, row_string in enumerate(lin_prog):
            tableau[row_idx], art_left = self.add_row(
                row_string, row_idx, art_left
            )

        # If there are any >= constraints, an additional objective is needed
        if self.geq_ids:
            art_row = self.create_art_row(tableau)
            tableau = np.vstack((art_row, tableau))
        return self.delete_empty(tableau)

    @staticmethod
    def generate_col_titles(*args):
        """Generate column titles for tableau of specific dimensions.

        args:
            n_vars : int
                Number of decision variables x1, x2...
            n_slack : int
                Number of slack variables added to turn inequalities into 
                equalities
            n_art_vars : int
                Number of artificial variables to be minimised

        returns:
            titles : list[str]
                Titles for each of the columns, x1, ..., s1, ..., a1, ..., RHS.
        """
        if len(args) != 3:
            raise ValueError("Must provide n_vars, n_slack, and n_art_vars")

        if not all(x >= 0 and isinstance(x, int) for x in args):
            raise ValueError("All arguments must be non-negative integers")

        # decision | slack | artificial
        string_starts = ["x", "s", "a"]
        titles = []
        for i in range(3):
            for j in range(args[i]):
                titles.append(string_starts[i] + str(j + 1))
        titles.append("RHS")
        return titles

    def find_pivot(self, tableau):
        """Finds the pivot row and column."""
        objective = self.objectives[-1]

        # Find entries of highest magnitude in objective rows
        sign = (objective == "min") - (objective == "max")
        col_idx = np.argmax(sign * tableau[0, : self.n_vars])

        # Choice is only valid if below 0 for maximise, and above for minimise
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

        # If element in pivot column is greater than zeron_stages, return
        # quotient or nan otherwise
        quotients = np.divide(dividend, divisor, out=nans, where=divisor > 0)

        # Arg of minimum quotient excluding the nan values. n_stages is added
        # to compensate for earlier exclusion of objective columns
        row_idx = np.nanargmin(quotients) + self.n_stages
        return row_idx, col_idx

    def pivot(self, tableau, row_idx, col_idx):
        """Pivots on value on the intersection of pivot row and column.

        args:
            tableau : np.ndarray
                Simplex tableau
            row_idx : int
                Pivot row index
            col_idx : int
                Pivot column index

        returns:
            tableau : np.ndarray
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

    def change_stage(self, tableau):
        """Exits first phase of the two-stage method by deleting artificial
        rows and columns, or completes the algorithm if exiting the standard
        case.

        args:
            tableau : np.ndarray
                Simplex tableau

        returns:
            tableau : np.ndarray
        """
        # Objective of original objective row remains
        self.objectives.pop()

        if not self.objectives:
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
        return tableau

    def run_simplex(self):
        """Operate on tableau until objective function cannot be
        improved further.

        returns:
            _ : dict[str, float] | dict[None, None]
                Variable values of the solution
        """
        # Stop simplex algorithm from cycling.
        for _ in range(100):
            # Completion of each stage removes an objective. If both stages
            # are complete, then no objectives are left
            if not self.objectives:
                self.col_titles = self.generate_col_titles(
                    self.n_vars, self.n_slack, self.n_art_vars
                )

                # Find the values of each variable at optimal solution
                return self.interpret_tableau(self.tableau, self.col_titles)

            row_idx, col_idx = self.find_pivot(self.tableau)

            # If there are no more negative values in objective row
            if self.stop_iter:
                # Delete artificial variable columns and rows. Update attributes
                self.tableau = self.change_stage(self.tableau)
            else:
                self.tableau = self.pivot(self.tableau, row_idx, col_idx)
        return {}

    def interpret_tableau(self, tableau, col_titles):
        """Given the final tableau, add the corresponding values of the basic
        decision variables to the `output_dict`

        args:
            tableau : np.ndarray
                Simplex tableau
            col_titles : list[str]
                Variable names for each column to be used for output dictionary

        returns:
            output_dict : dict[str, float]
                Values of decision variables and non-zero slack variables at
                optimal solution
        """
        # P = RHS of final tableau
        output_dict = {"P": abs(tableau[0, -1])}

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

        # Check for basic variables
        for title in col_titles:
            # Don't add RHS or slack variables to output dict
            if title[0] not in "R-s-a":
                output_dict.setdefault(title, 0)
        return output_dict
