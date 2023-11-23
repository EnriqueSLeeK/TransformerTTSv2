import sys


# Argument Parser
class ArgParser:
    def __init__(self):
        self.argument = {}

    def add_flag_argument(self, flag: str, default_argument: str):
        self.argument[flag] = default_argument

    def print_error_msg(self, message):
        print(message, file=sys.stderr)

    def parse(self):
        argv = sys.argv[1:]
        argv_len = len(argv)

        if argv_len == 0:
            print("Argv is empty defaulting config")
            return
        if argv_len % 2 != 0:
            self.print_error_msg(
                    "Odd number of argument, "
                    "cannot pair them, defaulting")
            return

        for pair_idx in range(0, argv_len, 2):
            if not argv[pair_idx] in self.argument:
                self.print_error_msg(
                        "The found flag is not registered, "
                        "terminating the script")
                sys.exit(1)
            self.argument[argv[pair_idx]] = argv[pair_idx + 1]

        def get_dict(self):
            return self.argument
