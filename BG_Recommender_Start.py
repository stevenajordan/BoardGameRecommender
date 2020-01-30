
# coding: utf-8

# In[ ]:


# Steven Jordan
# DSC 478 - Final Project
# Board Game Rec


# In[ ]:


def load_data():
    'Takes in no input, but file bgg.csv should be in the current directory'
    import pandas as pd
    import numpy as np
    import sklearn.preprocessing as skp
    
    # Load the file - should always be titled bgg.csv
    try:
        df = pd.read_csv('bgg.csv')
    except:
        print("Please ensure 'bgg.csv' file exists in the current directory")
    
    # Drop unnecessary columns
    try:
        data_and_labels = df.drop(['rank', 'game_id', 'min_time', 'max_time', 'year', 'avg_rating', 'num_votes', 'image_url', 'owned', 'designer'], axis = 1)
    
        # Remove games with missing numeric values
        data_and_labels = data_and_labels[data_and_labels.weight != 0]
        data_and_labels = data_and_labels[data_and_labels.min_players != 0]
        data_and_labels = data_and_labels[data_and_labels.max_players != 0]
        data_and_labels = data_and_labels[data_and_labels.avg_time != 0]
        data_and_labels = data_and_labels[data_and_labels.age != 0]
        data_and_labels = data_and_labels[data_and_labels.age != 42]
        data_and_labels = data_and_labels[data_and_labels.category != 'none']
        data_and_labels = data_and_labels[data_and_labels.mechanic != 'none']
    
        # Handles outliers in the data
        data_and_labels.loc[data_and_labels.min_players > 4, 'min_players'] = 4
        data_and_labels.loc[data_and_labels.max_players > 9, 'max_players'] = 9
        data_and_labels.loc[data_and_labels.avg_time > 205, 'avg_time'] = 205
        data_and_labels.loc[data_and_labels.weight > 4.5945, 'weight'] = 4.5945
        
        # Creates and adds dummy variables for the 'mechanic' and 'category' attributes
        mechanics_dummies = data_and_labels.mechanic.str.get_dummies(sep =', ')
        category_dummies = data_and_labels.category.str.get_dummies(sep =', ')
        data_and_labels_dummies = data_and_labels.drop(['mechanic', 'category'], axis = 1)
        data_and_labels_dummies = pd.concat([data_and_labels_dummies, mechanics_dummies, category_dummies], axis = 1)
        
        # Remove expansions as a category
        data_and_labels_dummies = data_and_labels_dummies[data_and_labels_dummies['Expansion for Base-game'] != 1]
        data_and_labels_dummies = data_and_labels_dummies.drop(['Expansion for Base-game'], axis = 1)
        
        # Splices the dataset into two dataframes that will be returned
        labels = data_and_labels_dummies.loc[:,['bgg_url','names','geek_rating']]
        data = data_and_labels_dummies.drop(['bgg_url', 'names','geek_rating'], axis = 1)
        
        # Scale the data
        min_max_scaler = skp.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(data)
        
        # Reset indices
        labels.reset_index(inplace = True)
    except:
        print("Please ensure 'bgg.csv' has the correct columns")
        
    return labels, data_scaled


# In[ ]:


def userfile(url, username):
    'Takes in a username as input, returns the favorited urls of the user'
    
    # Opens the user's file, or creates it if it doesn't exist
    try:
        uf = open(username + '.txt', 'r+')
        uf_list = uf.readlines()
        uf.close()
        
        clean_list = []
        for n in uf_list:
            clean_list.append(n.rstrip())
        
        if url in clean_list:
            return clean_list
        
        else:
            uf = open(username + '.txt', 'a+')
            uf.write(url)
            uf.write('\n')
            uf.close()
            clean_list.append(url)
            return clean_list
       
    except:
        print('\nNew username created.\n')
        uf = open(username + '.txt', 'w+')
        uf.write(url)
        uf.write('\n')
        uf.close()
        return [url]
    


# In[ ]:


def get_recs(url, username, top, num):
    '''Three input parameters is the board game url, the username, and whether the user wants to 
    select from the top 500 board games. There is no output parameter.'''
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.cluster import KMeans
    import sklearn.metrics.pairwise as skmp
    import warnings
    warnings.filterwarnings("ignore")
    
    # Call the function to get the existing labels and data
    labels, data = load_data()
    
    # Checks to make sure the input URL is valid
    if url not in np.array(labels.bgg_url):
        print('\nThat is not a valid URL in our dataset. Please try again.')
        return
        
    
    # Get the list of all favorites associated with this username
    faves = userfile(url, username)
    
    # Finds the indices of the favorited games
    fave_indices = []
    for game in faves:
        game_index = labels.index[labels['bgg_url'] == game][0]
        fave_indices.append(game_index)
    
    # Create an array of the attributes for the favorited games
    fave_games = []
    for i in fave_indices:
        fave_games.append(data[i])
    
    # Find the centroid of the favorited games
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=1).fit(fave_games)
    user_centroid = kmeans.cluster_centers_[0]
    
    # Find the similarity of all other games to the user centroid
    sim_list = []
    for game in data:
        sim = skmp.cosine_similarity([user_centroid], [game])
        sim_list.append(sim[0][0])
    
    # Zips the similarity scores to their labels
    tup = zip(sim_list, labels.names, labels.bgg_url, labels.geek_rating)
    tups_sorted = sorted(tup, key=lambda tup:tup[0], reverse = True)
    
    # Prints out the top recommendations
    print('Hello {}, your top {} board game recommendations are: \n'.format(username, num))
    count = 0
    for rec in tups_sorted:
        if top == 1:
            min_score = labels.geek_rating[500] #Sets the minimum rating if the user selected a top 500 game
            if (rec[2] not in faves) and rec[3] >= min_score: 
                print('Game Title: ', rec[1])
                print('URL: ',rec[2])
                print('Similarity Score', round(rec[0],4), '\n')
                count += 1
        elif top == 0:
            if rec[2] not in faves:
                print('Game Title: ', rec[1])
                print('URL: ',rec[2])
                print('Similarity Score', round(rec[0],4), '\n')
                count += 1
        if count == num:
            break
    
    


# In[ ]:


try:    
    while True:
        # Get user inputs
        username = input('Hello, what is your username? If new, please create one now. ')
        url = input('\nPlease paste here the URL of your favorited game from BoardGameGeek.com: ')
        while True:
            try:
                num = eval(input('\nHow many recommendations would you like to see? '))
                break
            except:
                print('Please type in a number')
                continue

        top = input('\nWould you like to limit your recommendations from the highest 500 rated board games? (Y/N) ')


        # Initiate recommender system
        if top in ['y', 'Y', 'yes', 'Yes', 'YES']:
            get_recs(url, username, 1, num)
            again = input('\nWould you like to add in another favorite? (Y/N) ')
            if again in ['y', 'Y', 'yes', 'Yes', 'YES']:
                continue
            else:
                input('\nThank you. Press Enter to Quit. ')
                break
        elif top in ['n', 'N', 'no', 'No', 'NO']:
            get_recs(url, username, 0, num)
            again = input('\nWould you like to add in another favorite? (Y/N) ')
            if again in ['y', 'Y', 'yes', 'Yes', 'YES']:
                continue
            else:
                input('\nThank you. Press Enter to Quit. ')
                break
        else:
            print('\nInvalid response. Please try again.')
except Exception as e: 
    print(e)
    input(' ')


# In[ ]:




