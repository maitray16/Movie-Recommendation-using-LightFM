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
   --Known positives:
        Toy Story (1995)
        Postino, Il (1994)
        Birdcage, The (1996)
   --Recommended:
        English Patient, The (1996)
        Sense and Sensibility (1995)
        Air Force One (1997)
User 2
   --Known positives:
        Return of the Jedi (1983)
        Event Horizon (1997)
        Schindler's List (1993)
   --Recommended:
        L.A. Confidential (1997)
        Jackie Brown (1997)
        Gattaca (1997)
User 3
   --Known positives:
        Seven (Se7en) (1995)
        Contact (1997)
        Starship Troopers (1997)
   --Recommended:
        Scream (1996)
        Game, The (1997)
        Contact (1997)
User 4
   --Known positives:
        Rumble in the Bronx (1995)
        Batman Forever (1995)
        To Wong Foo, Thanks for Everything! Julie Newmar (1995)
   --Recommended:
        Empire Strikes Back, The (1980)
        Raiders of the Lost Ark (1981)
        Star Wars (1977)
User 5
   --Known positives:
        Toy Story (1995)
        Babe (1995)
        Dead Man Walking (1995)
   --Recommended:
        Casablanca (1942)
        Lawrence of Arabia (1962)
        Graduate, The (1967)
'''
