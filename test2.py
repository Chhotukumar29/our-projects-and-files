class person:

    def __init__(self, name,surname,emailid, year_of_birth):
        self.name = name
        self.surname = surname
        self.emailid = emailid
        self.year_of_birth = year_of_birth
        
    def age(self,current_year):
        return current_year - self.year_of_birth

anuj_var = person("Anuj", "Bhandari", "anjubhandari@gmail.com",1994)
sudh = person("Sudhanshu", "Kumar", 'sudh@gmail.com', 1990)
gargi = person('Gargi', 'Goyal', 'gargigoyal@gmail.com', 1999)

print(anuj_var.name)
print(sudh.name)
print(gargi.name)

print(type(sudh))

print(anuj_var.name,anuj_var.surname)
print(anuj_var.age(2022))
