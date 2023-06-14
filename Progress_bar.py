import sys

# Adapted from stack overflow answer
class Progress_bar():
    def __init__(self, text, size):
        self.size = size
        self.text = text

        # setup toolbar
        sys.stdout.write(self.text + " [%s]" % (" " * self.size))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.size+1)) # return to start of line, after '['

    def next(self):
        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("]\n") # this ends the progress bar