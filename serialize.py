import pickle

serialized_data = pickle.dumps({'key': [[1.22323, 4.56789], [2.34567, 5.67890]], 'value': 'example'})
print(serialized_data)

with open('serialized_data.pkl', 'wb') as f:
    f.write(serialized_data)



with open('serialized_data.pkl', 'rb') as f:
    deserialized_data = pickle.load(f)
print(deserialized_data)