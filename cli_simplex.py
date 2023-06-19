from tableau import Tableau

EXAMPLE = r"""
Max: +2 +3 +1
+1 +1 +1 <= 40
+2 +1 -1 >= 10
+0 -1 +1 >= 10
"""


def cli():
    """Obtains user input via command line

    Returns:
        lin_prog : str
            multiline string containing the linear program
    """
    print(EXAMPLE, end="")
    print("Enter/Paste your content on separate lines as such. \n")
    lin_prog = []
    while True:
        row = input().strip()
        if row:
            lin_prog.append(row)
        else:
            break
    return lin_prog


def main():
    lin_prog = cli()
    t = Tableau(pre_lin_prog=lin_prog)
    output = t.run_simplex()
    for key, value in output.items():
        print(key, '=', value)


if __name__ == "__main__":
    main()
