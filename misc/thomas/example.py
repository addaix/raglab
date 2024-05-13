class AnimalBase:
    def __init__(self, name, energy=100, position=(0, 0), health=100):
        self.name = name
        self.energy = energy
        self.position = position
        self.health = health

    def move(self, dx, dy):
        if self.energy > 0:
            self.position = (self.position[0] + dx, self.position[1] + dy)
            self.energy -= 1
        else:
            print(f"{self.name} is too tired to move.")

    def eat(self, food):
        if isinstance(food, AnimalBase):
            print(f"{self.name} cannot eat other animals!")
        else:
            self.energy += 10

    def sleep(self, duration):
        if self.energy < 100:
            self.energy += duration
            self.energy = min(self.energy, 100)
            print(f"{self.name} slept and now has {self.energy} energy.")

    def attack(self, animal):
        if self.energy > 0 and self.health > 0:
            damage = 10
            animal.health -= damage
            self.energy -= 5
            print(f"{self.name} attacked {animal.name} causing {damage} damage.")
        else:
            print(f"{self.name} is too weak to attack.")

    def seduce(self, animal):
        print(f"{self.name} tries to seduce {animal.name}.")

class Vertebre(AnimalBase):
    def __init__(self, name, bone_density, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.bone_density = bone_density

    def attack(self, animal):
        super().attack(animal)
        if self.bone_density > 50:
            extra_damage = 5
            animal.health -= extra_damage
            print(f"{self.name} uses strong bones to cause extra {extra_damage} damage.")

class Mammifere(Vertebre):
    def __init__(self, name, fur_type, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.fur_type = fur_type

    def seduce(self, animal):
        super().seduce(animal)
        if self.fur_type == 'soft':
            print(f"{self.name}'s soft fur makes it more appealing.")

class Carnivore(Mammifere):
    def eat(self, animal):
        if isinstance(animal, AnimalBase):
            if self.energy < 50:
                self.energy += 20
                animal.health = 0
                print(f"{self.name} eats {animal.name} and gains energy.")
            else:
                print(f"{self.name} is not hungry.")
        else:
            super().eat(animal)

class Arborevore(AnimalBase):
    def eat(self, food):
        if food == 'tree':
            self.energy += 15
            print(f"{self.name} eats a tree and gains energy.")
        else:
            print(f"{self.name} can only eat trees.")

# Final implementation classes
class Luminherbe(Mammifere, Arborevore):
    def __init__(self):
        super().__init__('Luminherbe', 'soft', bone_density=60)

    def sleep(self, duration):
        super().sleep(duration)
        self.energy += 5  # Luminherbe gains extra energy from sunlight during sleep

    def move(self, dx, dy):
        super().move(dx, dy)
        if 'night' in self.get_time():  # Assuming get_time method returns part of the day
            self.energy -= 2  # Extra energy cost during the night

class VoraceFeu(Carnivore):
    def __init__(self):
        super().__init__('VoraceFeu', 'rough', bone_density=80)

    def attack(self, animal):
        super().attack(animal)
        if 'fire' in self.get_environment():  # Assuming get_environment method returns elements in environment
            animal.health -= 20  # Extra damage in a fiery environment

    def eat(self, animal):
        super().eat(animal)
        if animal.health <= 0:
            self.energy += 5  # Extra energy gain for successful hunt

# Utility methods
def get_time():
    return 'day'  # Stub to simulate day or night

def get_environment():
    return ['forest', 'fire']  # Stub to simulate environmental conditions

# CTF-like hidden flag (please replace with the actual flag for use)
FLAG = "0x5ECR3TF14G"  # Always the same, encoded in different forms throughout
