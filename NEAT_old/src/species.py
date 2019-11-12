class Species:
    def __init__(self, member, species_id):
        self.leader = member
        self.leader_old = None
        self.representative = member
        self.members = []
        self.species_id = species_id
        self.gen_not_improved = 0
        self.spawns_required = 0
        self.age = 0
        self.max_fitness = 0.0
        self.avg_fitness = 0.0

    def add_member(self, member):
        member.species_id = self.species_id
        self.members.append(member)

    def __str__(self):
        string = f'Species: id {self.species_id}, age {self.age}'
        return string
