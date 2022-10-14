class person:

    def __init__(self, name,surname,emailid, year_of_birth):
        self.name = name
        self.surname = surname
        self.emailid = emailid
        self.year_of_birth = year_of_birth

     def __init__(self,name,surname ):
         self.name = name
         self.surname = surname

    def age(self,current_year):
        return current_year - self.year_of_birth

anuj_var = person("Anuj", "Bhandari")
sudh = person("Sudhanshu", "Kumar")
gargi = person('Gargi', 'Goyal')

print(anuj_var.name)
print(sudh.name)
print(gargi.name)

print(type(sudh))

print(anuj_var.name,anuj_var.surname)
print(anuj_var.age(2022))
