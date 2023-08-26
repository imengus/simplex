import tkinter as tk
from tkinter import font
from tableau import Tableau


LABEL = """Overwrite the example to input your linear program

Note: '+2 +1 == 10' for '2*x + y = 10'
"""
EXAMPLE = r"""Max: +2 +3 +1
+1 +1 +1 <= 40
+2 +1 -1 >= 10
+0 -1 +1 >= 10
"""


class SimplexGui(tk.Tk):
    """
    User interface for Simplex algorithm
    """

    def __init__(self):
        """
        Initialise tkinter object by adding widgets
        """
        super().__init__()
        self.configure(background='white')
        # Dictionary containing variable value
        self.output = None
        self.input_str = None

        # Instructions
        self.in_label = tk.Label(self, text=LABEL, font=("Courier", 18), background="white")
        self.in_label.pack(padx=20, pady=20, fill="both")

        # Input box
        self.text = tk.Text(self, font=("gothic", 18), height=6, width=38)
        self.text.pack()
        self.text.insert("1.0", EXAMPLE)

        self.button_commit = tk.Button(
            self,
            height=1,
            width=10,
            text="Commit",
            background="white",
            font=("Courier", 18),
            command=lambda: self.simplex(),
        )
        self.button_commit.pack()

        # Runs simplex algorithm after input retrieved
        self.bind("<Return>", self.simplex)

    def display_output(self, output):
        """
        Add label containing output values
        """
        self.out_text = self.make_str(output)
        self.out_label = tk.Label(
            self, text=self.out_text, font=("Courier", 18), background="white")
        self.out_label.pack(padx=20, pady=20, fill="both")

    def simplex(self):
        """
        Process input and run simplex algorithm. Output results if algorithm
        terminates
        """
        # Obtain and process input
        self.input_str = self.text.get("1.0", "end-1c")

        # Generate tableau
        t = Tableau(pre_lin_prog=self.input_str.splitlines())
        output = t.run_simplex()
        self.display_output(output)

    @staticmethod
    def make_str(output_dict):
        """
        Combine variable output values into a single string.

        Args:
            output_dict : dict
                contains output values from algorithm

        Returns:
            string : str
                multiline string containing variable outputs
        """
        string = "Output:"
        for i, j in output_dict.items():
            string += f"\n {i} = " + str(j)
        return string


if __name__ == "__main__":
    simp = SimplexGui()
    simp.mainloop()
