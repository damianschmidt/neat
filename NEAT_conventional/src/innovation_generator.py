class InnovationGenerator:
    def __init__(self):
        self.current_innovation = 0

    def get_innovation(self):
        innovation_number = self.current_innovation
        self.current_innovation += 1
        return innovation_number
