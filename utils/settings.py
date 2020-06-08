def init():
    global settings_dict
    settings_dict = {}

def add(name, value):
    settings_dict[name] = value

def get(name):
    try:
        return settings_dict[name]
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 

def replace(name, new_value):
    try:
        settings_dict[name] = new_value
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 

def delete(name):
    try:
        del settings_dict[name]
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 

def remove_columns_names(name, excl_cols):
    try:
        setting = settings_dict[name]
    except KeyError:
        print("Setting {:s} not defined, please add it to the settings first".format(str(name))) 
    non_existing_cols = []
    for e in excl_cols:
        try:
            setting.remove(e)
        except ValueError:
            non_existing_cols.append(e)
    if len(non_existing_cols) > 0:
        string = "\n".join(non_existing_cols)
        print('The following columns were not removed from the setting "{:s}" as they were not existing:'.format(str(name)))
        print(string)

if __name__ == "__main__":
    init()
    add('test', list('abcdefg'))
    print(settings_dict)
    remove_columns_names('test', list('toiudgqyiazvd'))
