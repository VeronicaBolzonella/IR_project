import TruthTorchLM.long_form_generation.utils.safe_utils as utils
import pprint

serper_searcher = utils.SerperAPI(k=3)

query = "Who is Angelina Jolie?"
results=serper_searcher.run(query, k=3)

print(f"Results{results}")
print(f"Type: {type(results)}")
print("\n")
print(f"Attributes (if an object): {dir(results)}")
print("\n")
pprint.pprint(f"For nested dictionaries: {results}")
print("\n")
try:
    print(dict(results))
except:
    pass
print("\n")
print(f"Raw String: {str(results)}")
# import inspect
# import TruthTorchLM.long_form_generation.utils.safe_utils as utils

# # 1) Inspect the class
# print(utils.SerperAPI)

# # 2) Inspect the source of the class
# print(inspect.getsource(utils.SerperAPI))

# # 3) Inspect only the run() method
# print(inspect.getsource(utils.SerperAPI.run))