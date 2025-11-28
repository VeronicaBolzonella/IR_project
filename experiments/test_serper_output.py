import os

import TruthTorchLM.long_form_generation.utils.safe_utils as utils
import pprint

# Set Serper API key like this: SERPER_API_KEY

print("Initialising serper searcher")
serper_searcher = utils.SerperAPI(k=3)

query = "Who is Angelina Jolie?"
print("Running serper searcher...")
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
