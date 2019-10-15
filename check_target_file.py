
fileout = open("regions_for_learning_with_head.clean.equal_size.bed","w")

with open("regions_for_learning_with_head.clean.bed") as f:
    for line in f:
        elements = line.strip().split("\t")
        if elements[0] == "CHR":
            fileout.write(line)
            continue
        if(  int(elements[2]) - int(elements[1])  ) == 1500:
            fileout.write(line)