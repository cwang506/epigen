from xml.dom import minidom

def generate_model_xml(num_disease_snps, num_non_disease_snps, path_to_file):
    root = minidom.Document()
    xml = root.createElement('Parameterized Model')
    xml.setAttribute("size", str(num_disease_snps + num_non_disease_snps))
    xml.setAttribute("phenotype", "quantitative")
    xml.setAttribute("stdev", "1.3")
    root.appendChild(xml)

    baseline = root.createElement("BaselineModel")
    baseline.setAttribute("alpha", "15")
    xml.appendChild(baseline)
    for i in range(num_disease_snps):
        marginalmodel = root.createElement("MarginalModel")
        marginalmodel.setAttribute("alpha", "2")
        marginalmodel.setAttribute("type", "additive")
        xml.appendChild(marginalmodel)
        pos = root.createElement("pos")
        index = root.createTextNode(str(i))
        pos.appendChild(index)
        marginalmodel.appendChild(pos)
    xml_str = root.toprettyxml(indent='\t')
    with open(path_to_file, 'w') as f:
        f.write(xml_str)   
    f.close()

