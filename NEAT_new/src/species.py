class Species:
    def __init__(self, species_id, generation):
        self.species_id = species_id
        self.members = {}
        self.created_generation = generation
        self.last_improved = generation
        self.mascot = None
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

    def updated_species(self, mascot, members):
        self.mascot = mascot
        self.members = members

    def get_members_fitness(self):
        members_fitness = [member.fitness for member in self.members.values()]
        return members_fitness

# SpeciesSet?
