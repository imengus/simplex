# Simplex
Python implementations of the simplex algorithm for solving linear programs in tabular form with
- `>=`, `<=`, and `=` constraints and
- each variable `x1, x2, ...>= 0`.

## Contents:
|Name | Purpose |
| ----------------- | ----------------------------------- |
| cli_simplex.py | Takes linear program inputs from command line |
| gui_simplex/ | Takes inputs using a Tkinter GUI |
| simplex.py | Doctested core algorithm |

# Executing the simplex algorithm

## Maximising:
The algorithm has steps for maximising as follows:
1. Find the most negative value in the `x1, x2, ...` columns. This is the pivot row.
2. Divide the RHS by the corresponding x value (i.e. the value of the variable for the pivot column at a particular row) for each constraint row to get the quotients.
3. Choose the x value which produces the lowest positive quotient. The row containing this x value is the pivot row, and this x value becomes the pivot value.
4. Multiply each value in the pivot row by the reciprocal of the pivot value. The x value that was initially the pivot value should be 1.
5. Add the entire row multiplied by the negatives of the other x values in the pivot column, including the objective row. The pivot column should now only contain 0s except for a 1 entry in the pivot row.
6. Check if there are any more negative values in the `x1, x2, ...` columns. If so, repeat at step 1. If not, terminate the algorithm. 

## Minimising:
For minimising, the process is the same but the most positive value is sought and the algorithm terminates when there are no positive values left in top, or objective, row. 

# Converting linear programs to simplex tableaus

The constraints of a linear program define an n-dimensional feasible region, where n = number of variables. Any point within this region will satisfy the constraints, but we are interested in the coordinates of a point at which  an objective function P is maximised (or minimised.)

The simplex algorithm starts at basic solution `x1 = x2 = ... = 0`, then checks  each vertex of the feasible region in the direction that would increase (or decrease if minimise) P the fastest. When objective value P cannot be improved any further, i.e. the optimum vertex is found, the algorithm terminates.

This simplex algorithm operates on tableaus (matrices). The steps for converting an example linear program into a simplex tableau are shown:
```
    Minimise P = 8x1 + 6x2 
    subject to:   x1 +  x2 <= 20
                  x1       >= 5
                        x2  = 6
```
1. Turn objective into Maximise and move variables to left-hand side:

    Minimise P == Maximise -P.
    Then, `-P = 8x1 + 6x2  == P + 8x1 + 6x2 = 0`

2. Add slack variables to make <= constraint equal to right-hand side (while x1 and x2 are in feasible region):

    `x1 + x2 <= 20  == x1 + x2 + s1 = 20`, where s1 is a non-negative slack variable that ensures `x1 + x2` (while less than 20) is equal to 20. E.g. at `x1 = x2 = 0, s1 = 20`; whereas at `x1 = x2 = 20, s1 = 0`.

3. Subtract surplus (equivalent to slack) variables and add artificial variables to make `>=` constraint equal to right-hand side (again, for x1 and x2 values within feasible region):
    `x1 >= 5 == x1 - s2 + a1 = 5`, where s2 and a1 are non-negative surplus and artificial variables, respectively. The basic solution `x1 = x2 = 0` is not
    in the feasible region for linear programs with `>=` constraints. An artificial variable is added to satisfy equality when the coordinates of x1 and x2 lie outside the feasible region.

4. Split equality constraints into `>=` and `<=` constraints, then repeat 2. and 3.

After these steps are completed, our linear program will look like:
```
    Maximise P + 8x1 + 6x2                                = 0
    subject to:   x1 +  x2 + s1                           = 20
                  x1            - s2            + a1      = 5
                        x2           + s3                 = 6
                        x2                - s4       + a2 = 6
```
The simplex algorithm still cannot begin, given that `x1 = x2 = 0` is not a basic solution as the origin lies outside the feasible region. Jumping to a vertex on the feasible region would involve minimising the artificial variables to 0, which constitutes the 5th step. This creates another objective, `Minimise Q = a1 + a2`. 

Adding the formerly `>=` rows of our linear program, we get:
```
    (   + x2      - s4      + a2 = 6) + 
    (x1      - s2      + a1      = 5) =
     x1 + x2 - s2 - s4 + a1 + a2 = 11
```
Which equals to `x1 + x2 - s2 - s4 + Q = 11` given that `Q = a1 + a2`.

Now our linear program will look like
```
1.  Minimise Q +  x1 +  x2      - s2      - s4            = 11
2.  Maximise P + 8x1 + 6x2                                = 0
    subject to:   x1 +  x2 + s1                           = 20
                  x1            - s2            + a1      = 5
                        x2           + s3                 = 6
                        x2                - s4       + a2 = 6
```
This linear program is two-stage, as we must first minimise the artificial variables before we can maximise `Q`, by moving across the vertices of the 2d feasible region.

Now comes the tabular representation so that this code can run. Each entry in tableau represents the coefficient of the variables in the columns, for each constraint row:
```
[[Q   P   x1  x2  s1  s2  s3  s4  a1  a2  RHS]
 [ 1.  0.  1.  1.  0. -1.  0. -1.  0.  0. 11.]
 [ 0.  1.  8.  6.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  1.  1.  1.  0.  0.  0.  0.  0. 20.]
 [ 0.  0.  1.  0.  0. -1.  0.  0.  1.  0.  5.]
 [ 0.  0.  0.  1.  0.  0.  1.  0.  0.  0.  6.]
 [ 0.  0.  0.  1.  0.  0.  0. -1.  0.  1.  6.]]
```
The leftmost two columns for `P` and `Q` have no impact on the outcome of the algorithm, so they are excluded from the tableau for simplicity:

The tableau is then inputted into the simplex Python program as the numpy array:
```
tableau = np.array([[ 1.  1.  0. -1.  0. -1.  0.  0. 11.],
                    [ 8.  6.  0.  0.  0.  0.  0.  0.  0.],
                    [ 1.  1.  1.  0.  0.  0.  0.  0. 20.],
                    [ 1.  0.  0. -1.  0.  0.  1.  0.  5.],
                    [ 0.  1.  0.  0.  1.  0.  0.  0.  6.],
                    [ 0.  1.  0.  0.  0. -1.  0.  1.  6.]])
```
with `n_vars = 2`

If there were no `>=` or `=` constraints, there would be no need to mimimise artificial variables and only the row for P would be included to produce a standard simplex tableau.
"""