class Animal:

    def __init__(self, name, type):
        self.name = name
        self.type = type


class Dog(Animal):

    def __init__(self,name,type):
        super().__init__(name, type)
        self.name = name
        self.type = type

    def getName():
        return self.name

    def getType():
        return self.type


Ankit = Dog()