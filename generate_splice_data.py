#!/usr/bin/python
"""
generate data set
"""

from collections import defaultdict
import os
import helper
import random



def create_taxonomy():
    
    from task_similarities import TreeNode
    root = TreeNode("root")
    chordata = TreeNode("chordata")
    protostomia = TreeNode("protostomia")
    root.add_child(chordata)
    root.add_child(protostomia)
    c_savignyi = TreeNode("c_savignyi")
    chordata.add_child(c_savignyi)
    vertebrata = TreeNode("vertebrata")
    chordata.add_child(vertebrata)
    actinopterygii = TreeNode("actinopterygii")
    vertebrata.add_child(actinopterygii)
    d_rerio = TreeNode("d_rerio")
    actinopterygii.add_child(d_rerio)
    g_aculeatus = TreeNode("g_aculeatus")
    actinopterygii.add_child(g_aculeatus)
    t_nigroviridis = TreeNode("t_nigroviridis")
    actinopterygii.add_child(t_nigroviridis)
    aves = TreeNode("aves")
    vertebrata.add_child(aves)
    g_gallus = TreeNode("g_gallus")
    aves.add_child(g_gallus)
    m_gallopavo = TreeNode("m_gallopavo")
    aves.add_child(m_gallopavo)
    mammals = TreeNode("mammals")
    vertebrata.add_child(mammals)
    b_taurus = TreeNode("b_taurus")
    mammals.add_child(b_taurus)
    h_sapiens = TreeNode("h_sapiens")
    mammals.add_child(h_sapiens)
    m_musculus = TreeNode("m_musculus")
    mammals.add_child(m_musculus)
    protostomia.children
    c_elegans = TreeNode("c_elegans")
    protostomia.add_child(c_elegans)
    d_melanogaster = TreeNode("d_melanogaster")
    protostomia.add_child(d_melanogaster)
    root.plot()


def get_positions_GFT(file_name, chromosome_names):
    '''
    generate list of acceptor positions
    '''
    
    positions = defaultdict(list)
    
    f = file(file_name)
    
    for line in f:
       
        if 'exon_number "1"' in line:
            continue

        tokens = line.strip().split("\t")
        
        #print line, len(tokens), tokens
        
        # fetch only acceptor splice sites  EEE[don]IIII...IIII[acc]EEE
        if tokens[2]=="exon" and tokens[6] == "+":
            
            if chromosome_names==None or tokens[0] in chromosome_names:
                positions[tokens[0]].append(int(tokens[3]))

    f.close()
    
    return positions



class GenomeHandler:
    
    
    def __init__(self, fasta_fn, chr_list):
        
        self.fasta_fn = fasta_fn
        self.chr_list = chr_list
        self.seqs = {}

        from Bio import SeqIO
        seq_io = SeqIO.parse(file(fasta_fn), "fasta")
        
        for record in seq_io:
            
            if record.id in chr_list:
                
                print "loading chromosome %s" % (record.id)
                
                self.seqs[record.id] = record

        self.genome_length = sum([float(self.get_length(c)) for c in self.chr_list])
                        

    def get_codon(self, chr_name, pos):
        
        assert (chr_name in self.chr_list)
        
        return self.seqs[chr_name].seq[pos-1:pos-1 + 3]


    def get_kmer(self, chr_name, pos, k):
        
        assert (chr_name in self.chr_list)
        
        return self.seqs[chr_name].seq[pos:pos + k]
        

    def get_window(self, chr_name, pos):
                       
        return self.seqs[chr_name].seq[pos-100:pos+100]
    
    
    def get_length(self, chr_name):
        
        return len(self.seqs[chr_name])


    def get_hits(self, chr_name, n):
        """
        if we sample n times from genome, 
        how many times do we draw from chromosome chr_name
        """

        return int((float(self.get_length(chr_name)) / self.genome_length) * n)



def get_chr_names(org_name):
    

    chr_names = None    
    
    if org_name == "b_taurus":        
        chr_names = [str(i) for i in range(1, 30)]
        chr_names.append("X")
        chr_names.append("Y")
 
    if org_name == "c_elegans":       
        chr_names = ["I", "II", "III", "IV", "V"]
 
    if org_name == "d_melanogaster": 
        chr_names = ['3RHet', '2R', '3R', '2RHet', '3LHet', '2LHet', '4', '3L', '2L']
        
    if org_name == "m_musculus":        
        chr_names = [str(i) for i in range(1, 20)]  
        chr_names.append("X")
        chr_names.append("Y")
        
    if org_name == "h_sapiens":
        chr_names = [str(i) for i in range(1, 22)]
        chr_names.append("X")
        chr_names.append("Y")
        
    return chr_names



def create_seq_data(org_name, work_dir):
    '''
    sample data
    '''

    print "processing organism", org_name

    files = os.listdir(work_dir)
       
    fn_seq = work_dir + [fn for fn in files if fn.endswith(".fa")][0] 
    fn_pos = work_dir + [fn for fn in files if fn.endswith(".gtf")][0]
        
    chr_names = get_chr_names(org_name)

    max_mismatches = 2

    print "loading positions"

    # load positions
    positions = get_positions_GFT(fn_pos, chr_names)
    
    chr_names = positions.keys()
    
    print "done with positions" 
    
    print "chromosomes", chr_names

    genome = GenomeHandler(fn_seq, chr_names)

    
    pos_seqs = []
    neg_seqs = []
 
    num_pos = 0
 

    for chr in chr_names:
        
        print "processing chromosome %s" % (chr) 

        # assemble positive list
        false_positions = set()
        
        # positive positions
        pos_pos = []

        num_non_consensus = 0
        num_many_ns = 0

        for pos in positions[chr]:


            if genome.get_kmer(chr, pos-3, 2).count("AG") != 1:
                false_positions.add(pos)
                num_non_consensus += 1
            else:
                if genome.get_window(chr, pos).count("N") < max_mismatches:
                    pos_pos.append(pos)
                else:
                    num_many_ns += 1
                    #print "discarding candidate because of %i mismatches, current len=%i" % (genome.get_window(chr, pos).count("N"), len(pos_seqs))
    
        print "%s: keeping %i/%i positive position, where %i were non-consensus and %i had more than %i Ns" % (chr, len(pos_pos), len(positions[chr]), num_non_consensus, num_many_ns, max_mismatches)

        if num_pos > 0:

            # subsample
            num_hits = genome.get_hits(chr, num_pos)

            # take no more than possible:
            num_hits = min(num_hits, len(pos_pos))

            print "subsampling %i sequences from %i positive sequence from chr %s" % (num_hits, len(pos_pos), chr)
            for pos in random.sample(pos_pos, num_hits):
                pos_seqs.append(genome.get_window(chr, pos).tostring().replace("N", "A"))
        else:
            # take everything
            print "using all %i sequences from chr %s" % (len(pos_pos), chr)
            for pos in pos_pos:
                pos_seqs.append(genome.get_window(chr, pos).tostring().replace("N", "A"))

 
    print "done with processing of positives for", org_name

    num_neg = len(pos_seqs) * 10


    # generate negative list

    # sample from chromosomes
    for chr_name in genome.chr_list:

        print "processing %s to generate negative examples" % (chr_name)

        # how much space to leave at the end of the chromosome
        margin = 100000
        end = genome.get_length(chr_name) - margin

        # sample fraction from position list
        num_hits = genome.get_hits(chr_name, num_neg)

        neg_pos = set()

        # sample from genome
        while len(neg_pos) < num_hits:

            i = random.randint(margin, end)

            # discard if non-consensus OR too many Ns OR positive
            if genome.get_kmer(chr_name, i-3, 2).count("AG") == 1 and \
               genome.get_window(chr_name, i).count("N") < max_mismatches and \
               not i in positions[chr_name]:

                neg_pos.add(i)
                neg_seqs.append(genome.get_window(chr_name, i).tostring().replace("N", "A"))
 

    print "done with processing of negatives for", org_name

    return (neg_seqs, pos_seqs) 
       


def main():

    base_dir = "data/splice"
    organisms = os.listdir(base_dir)    
        
        
    for org_name in organisms:
    
        print "processing", org_name

        work_dir = base_dir + org_name + "/"
        
        (neg, pos) = create_seq_data(org_name, work_dir)
        
        result = {}
        result["pos"] = pos
        result["neg"] = neg

        print "======================="
        print "%s pos=%i, neg=%i" % (org_name, len(pos), len(neg))

        save_fn = work_dir + "seqs_acc.pickle"
        
        helper.save(save_fn, result)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
