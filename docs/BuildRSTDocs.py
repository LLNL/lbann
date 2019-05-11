#from RSTDocsFlavorText import *

import xml.etree.ElementTree as etree
import os, runpy

rst_docs_globals = runpy.run_path("RSTDocsFlavorText.py")
lbann_rst_headers = rst_docs_globals["lbann_rst_headers"]
lbann_rst_flavor_text = rst_docs_globals["lbann_rst_flavor_text"]

# Some globals cuz lazy
xml_root_dir = 'doxy_out/xml/'

def strip_template(class_name):
    ind = class_name.find('<')
    if ind > 0:
        return class_name[0:ind]
    return class_name

# This will return a list of length 2, [longest_namespace, class_name]
# E.g., split_namespace("A::B::C::myclass") will return ["A::B::C","myclass"].
#
# If no namespace, the first entry will be empty string.
def split_namespace(class_name):
    ind = class_name.rfind(':')
    if ind > 0:
        return class_name[0:ind-1], class_name[ind+1:]
    return "", class_name

def strip_namespace(class_name):
    ind = class_name.rfind(':')
    if ind > 0:
        return class_name[ind+1:]
    return class_name

def get_known_subdirs(topdir, all_dirs):
    subdirs = []
    for d in all_dirs:
        if d == topdir: continue
        commonprefix = os.path.commonprefix([topdir, d])
        if commonprefix == topdir:
            if os.path.dirname(d) == topdir:
                subdirs.append(d)
    return subdirs

def is_abstract_class_from_element(class_element):
    abstract = class_element.get('abstract')
    if abstract is None or abstract == 'no':
        return False
    return abstract == 'yes'

def is_abstract_class(class_name, xml_file):
    class_tree = etree.parse(xml_file)
    class_root = class_tree.getroot()

    class_element = class_root.find('compounddef')
    if class_element.findtext('compoundname') != class_name:
        raise Exception('bad compoundname')
    return is_abstract_class_from_element(class_element)

def is_base_class_from_element(class_element):
    base_element = class_element.find('basecompoundref')
    if base_element is not None:
        return False
    return True

def is_base_class(class_name, xml_file):
    class_tree = etree.parse(xml_file)
    class_root = class_tree.getroot()

    class_element = class_root.find('compounddef')
    if class_element.findtext('compoundname') != class_name:
        raise Exception('bad compoundname')
    return is_base_class_from_element(class_element)

def get_class_directory_from_element(class_element):
    loc = class_element.find('location')
    if loc is None:
        raise Exception("Class has no location")
    filename = loc.get('bodyfile')
    if filename is None:
        filename = loc.get('file')
        if filename is None:
            raise Exception("No file or bodyfile in location")
    return os.path.dirname(filename)

def get_class_directory(class_name, xml_file):
    class_tree = etree.parse(xml_file)
    class_root = class_tree.getroot()
    class_name = strip_template(class_name)

    class_element = class_root.find('compounddef')
    if class_element.findtext('compoundname') != class_name:
        raise Exception('compoundname "' + class_element.findtext('compoundname') + '" does not match "' + class_name + '"')
    return get_class_directory_from_element(class_element)

def is_base_class_rel_to_dir(class_name, xml_file):
    class_tree = etree.parse(xml_file)
    class_root = class_tree.getroot()

    class_element = class_root.find('compounddef')
    if class_element.findtext('compoundname') != class_name:
        raise Exception('bad compoundname')

    this_class_dir = get_class_directory_from_element(class_element)
    base_element = class_element.find('basecompoundref')

    if base_element is None:
        return True

    base_name = base_element.text
    base_class_refid = base_element.get('refid')
    if base_class_refid is None: # Base class not found
        return True

    base_class_xml = os.path.join(
        xml_root_dir,base_class_refid+'.xml')
    base_class_dir = get_class_directory(base_name, base_class_xml)

    return base_class_dir != this_class_dir

def is_public_class_from_element(class_element):
    return class_element.get('prot') == 'public'

def is_public_class(class_name, xml_file):
    class_tree = etree.parse(xml_file)
    class_root = class_tree.getroot()

    class_element = class_root.find('compounddef')
    if class_element.findtext('compoundname') != class_name:
        raise Exception('bad compoundname')

    return is_public_class_from_element(class_element)

# Write a simple RST file for a class
def write_class_rst_file(class_name, breathe_project_name, *args, **kwargs):
    namespace = kwargs.get('namespace', '')
    display_name = kwargs.get('display_name', class_name)
    description = kwargs.get('description', '')
    header_string = kwargs.get('header_string', '')
    output_dir = kwargs.get('output_dir', os.getcwd())
    output_filename = kwargs.get('output_filename', '')
    subclasses = kwargs.get('subclasses', {})

    # Handle defaults more rigorously
    if namespace == '':
        namespace, class_name = split_namespace(class_name)
    if not namespace == '':
        namespace = namespace + '::'

    # Possibly rebuild the structure since breathe needs namespace
    # information
    full_class_name = namespace + class_name

    if output_filename == '':
        output_filename = class_name + '.rst'

    if header_string == '':
        header_string = "Documentation of "+display_name

    equal_string = '=' * (len(header_string) + 5)

    output_file = os.path.join(output_dir,output_filename)
    with open(output_file, 'w') as f:
        f.write(header_string + '\n')
        f.write(equal_string + '\n\n')
        if description != '':
            f.write(description + '\n\n')
        else:
            f.write('\n')
        f.write('.. doxygenclass:: ' + full_class_name + '\n')
        f.write('    :project: ' + breathe_project_name + '\n')
        f.write('    :members:\n\n')
        if len(subclasses) > 0:
            f.write('.. toctree::\n')
            f.write('    :maxdepth: 1\n')
            f.write('    :caption: Derived Classes\n\n')
            for sc, sc_out_dir in subclasses.items():
                sc_no_ns = strip_namespace(sc)
                if sc_out_dir == output_dir:
                    sc_rst_path = sc_no_ns
                else:
                    sc_rst_path = os.path.join(
                        os.path.relpath(sc_out_dir, output_dir),
                        sc_no_ns);
                f.write('    ' + sc_no_ns + ' <' + sc_rst_path + '>\n')
    return

# Adds things from rhs into lhs. Keys are anything, values are lists
def merge_dir_class_maps(lhs, rhs):
    for d, cls in rhs.items():
        if d not in lhs:
            lhs[d] = cls
        else:
            lhs[d] += cls

# Writes a file called "strip_namespace(class_name).rst" in "output_dir"
def process_class(class_name, xml_file, output_root_dir):

    # Get the XML tree for this class
    class_tree = etree.parse(xml_file)
    class_root = class_tree.getroot()

    # Get the description of this class
    compounds_in_file = class_root.findall('compounddef')
    if len(compounds_in_file) > 1:
        raise Exception("Found multiple compounds in file: "+xml_file)

    compound = compounds_in_file[0]

    # Ensure there's nothing funky in the file.
    if compound.findtext('compoundname') != class_name:
        raise Exception("Found unexpected compounddef \""
                        + compound.findtext('compoundname')
                        + "\" in file: " + xml_file)
    if compound.get('kind') != "class":
        raise Exception("\"" + class_name + "\" does not have kind=\"class\". "
                        + "File: " + xml_file)

    # Build the output directory path
    class_dir = get_class_directory_from_element(compound)
    output_dir = os.path.relpath(class_dir, "../include/lbann/")
    # Add the base prefix
    file_output_dir = os.path.normpath(
        os.path.join(output_root_dir, output_dir))
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    # Build output for all derived classes
    subclasses = {}
    output_dir_class_map = {}
    for derived in compound.iter('derivedcompoundref'):
        derived_name = strip_template(derived.text)
        derived_xml = os.path.join(xml_root_dir,
                                   derived.get('refid') + ".xml")

        sc_out_dir, sc_dir_class_map = process_class(
            derived_name, derived_xml, output_root_dir)

        merge_dir_class_maps(output_dir_class_map, sc_dir_class_map)
        subclasses[derived_name] = sc_out_dir

    # Write the RST for this class
    header_string = "Documentation of " + class_name
    write_class_rst_file(class_name, "lbann",
                         output_dir=file_output_dir,
                         subclasses=subclasses)

    # Add this class to the map
    if file_output_dir not in output_dir_class_map:
        output_dir_class_map[file_output_dir] = [class_name]
    else:
        output_dir_class_map[file_output_dir].append(class_name)

    return file_output_dir, output_dir_class_map

#
# Actual code starts here
# Let's see if I can write everything
#

# Set the XML output directory relative to this directory
xml_root_dir = 'doxy_out/xml/'
index_tree = etree.parse(xml_root_dir + 'index.xml')
index_root = index_tree.getroot()

# Set the RST output directory relative to this directory
rst_base_dir = "lbann"

# Find all classes in the index
class_to_file_map = {}
for neighbor in index_root.iter('compound'):
    if neighbor.get('kind') == 'class':
        class_to_file_map[neighbor.findtext('name')] \
            = os.path.join(xml_root_dir,neighbor.get('refid') + '.xml')

# Build all of the class documentation
dir_all_class_map = {}
for cls, fn in class_to_file_map.items():
    if is_base_class(cls, fn) and is_public_class(cls, fn):
        out_dir, all_out_dirs = process_class(cls, fn, rst_base_dir)
        merge_dir_class_maps(dir_all_class_map, all_out_dirs)

# Write the high-level files, one file per directory except where
# noted below
ignore_dirs = [ os.path.join(rst_base_dir,d) for d in
                ['data_distributions','utils/impl']]

# Remove the ignored dirs from the map
for d in ignore_dirs:
    if d in dir_all_class_map:
        del dir_all_class_map[d]

all_dirs = list(dir_all_class_map.keys())
for d in all_dirs:
    dir_without_base = os.path.relpath(d, rst_base_dir)
    if dir_without_base in lbann_rst_headers:
        header_string = lbann_rst_headers[dir_without_base]
    else:
        header_string = os.path.basename(dir_without_base)

    equal_string = '=' * (len(header_string) + 5)

    if dir_without_base in lbann_rst_flavor_text:
        flavor_text = lbann_rst_flavor_text[dir_without_base]
    else:
        flavor_text = None

    abstract_classes = []
    concrete_classes = []
    for c in dir_all_class_map[d]:
        if is_abstract_class(c, class_to_file_map[c]):
            abstract_classes.append(strip_namespace(c))
        else:
            concrete_classes.append(strip_namespace(c))

    abstract_classes.sort()
    concrete_classes.sort()

    subdirs = [os.path.basename(d) for d in get_known_subdirs(d, all_dirs)]
    subdirs.sort()

    if dir_without_base == '.':
        filename = os.path.join(rst_base_dir, "lbann.rst")
    else:
        filename = os.path.join(d,os.path.basename(d)+'_dir.rst')

    with open(filename, 'w') as f:
        f.write(header_string+'\n')
        f.write(equal_string+'\n\n')
        if flavor_text is not None:
            f.write(flavor_text+'\n')

        if len(abstract_classes) > 0:
            f.write('\n')
            f.write('.. toctree::'+'\n')
            f.write('    :maxdepth: 1'+'\n')
            f.write('    :caption: Abstract Classes\n\n')
            for cls in abstract_classes:
                f.write('    class '+cls+' <'+cls+'>\n')

        if len(concrete_classes) > 0:
            f.write('\n')
            f.write('.. toctree::'+'\n')
            f.write('    :maxdepth: 1'+'\n')
            f.write('    :caption: Concrete Classes\n\n')
            for cls in concrete_classes:
                f.write('    class '+cls+' <'+cls+'>\n')

        if len(subdirs) > 0:
            f.write('\n')
            f.write('.. toctree::'+'\n')
            f.write('    :maxdepth: 1'+'\n')
            f.write('    :caption: Subdirectories\n\n')
            for sdir in subdirs:
                f.write('    '+sdir+'/'+sdir+'_dir\n')
