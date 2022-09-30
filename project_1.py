import random
# snake water gun or Rock paper Scissors

def gameWin(comp, you):
    #IF two values are equal, declare a tie!
    if comp==you:
        return None
    # Check for all possibilities computer chose s
    elif comp=='s':
        if you=='w':
            return False
        elif you=='g':
            return True
    # Check for all possibilities computer chose w
        elif comp == 'w':
            if you == 's':
                return False
            elif you == 's':
                return True
    # Check for all possibilities computer chose g
            elif comp == 'g':
                if you == 's':
                    return False
                elif you == 'w':
                    return True

print("Comp Turn: Snake(s) Water(w) or Gun(g)?")
randNo= random.randint(1,3)
if randNo==1:
    comp='s'
elif randNo==2:
    comp='w'
elif randNo==3:
    comp ='g'

you=input("Your Turn: Snake(s) Water(w) or Gun(g)?")
a=gameWin(comp, you)
print(f"Computer chose {comp}")
print(f"you chose {you}")
if a == None:
    print("The game is tie!")
elif a:
    print("You Win")
else:
    print("You Lose!")

