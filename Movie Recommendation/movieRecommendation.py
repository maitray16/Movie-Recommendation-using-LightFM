import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("   --Known Movies:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("   --Recommended Movies:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


'''
OUTPUT

User 1
   --Known Movies:
        Toy Story (1995)
        Postino, Il (1994)
        Birdcage, The (1996)
   --Recommended Movies:
        English Patient, The (1996)
        Contact (1997)
        L.A. Confidential (1997)
User 2
   --Known Movies:
        Return of the Jedi (1983)
        Event Horizon (1997)
        Schindler's List (1993)
   --Recommended Movies:
        Contact (1997)
        Scream (1996)
        L.A. Confidential (1997)
User 3
   --Known Movies:
        Seven (Se7en) (1995)
        Contact (1997)
        Starship Troopers (1997)
   --Recommended Movies:
        Scream (1996)
        Game, The (1997)
        Air Force One (1997)
User 4
   --Known Movies:
        Rumble in the Bronx (1995)
        Batman Forever (1995)
        To Wong Foo, Thanks for Everything! Julie Newmar (1995)
   --Recommended Movies:
        Raiders of the Lost Ark (1981)
        Monty Python and the Holy Grail (1974)
        Terminator, The (1984)
User 5
   --Known Movies:
        Toy Story (1995)
        Babe (1995)
        Dead Man Walking (1995)
   --Recommended Movies:
        Godfather, The (1972)
        Rear Window (1954)
        Annie Hall (1977)
'''
