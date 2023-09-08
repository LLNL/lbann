import spack.environment as ev
import itertools

# Grab the environment
x = ev.active_environment()
# Find all of the top level concretized specs
y = list(x.concretized_specs())
roots = []
# Search the top level root packages
for i in y:
    # Separate LBANN from the rest of the roots
    if 'lbann' in i[1].format('{name}/{version}-{hash:7}'):
        lbann = i
    else: # Keeep the rest in a single list
        roots.append(i)

#print(lbann[1].format('{name}/{version}-{hash:7}'))

# Get the module names and hashes of the roots
module_list = []
for i in roots:
    module = i[1].format('{name}/{version}-{hash:7}')
    if module not in module_list:
        module_list.append(module)

# Find the dependent modules of LBANN
deps = list(itertools.chain(lbann[1].dependencies()))
for i in deps:
    module = i.format('{name}/{version}-{hash:7}')
    if module not in module_list:
        module_list.append(module)

# module_list = root_modules
# module_list.extend(dependent_modules)
#print(module_list)
print(";".join(module_list))
# print('\n')
# print("; ".join(sorted(set(module_list))))
# print('\n')
