def connectedTimebins(time):
    if time < 250:
        return 1 
    elif time < 500:
        return 2
    elif time < 750:
        return 3
    elif time < 1000:
        return 4
    elif time > 1000:
        return 5
    else:
        return 6
    


