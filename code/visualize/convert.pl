# Convert the output of linearize.lua into a JSON data structure

use JSON;

my @prelus;
my @edges;
my @layeredges;
my $layer = -1;
my $out = 0;
while(<STDIN>) {
   next if /^\s*$/; 
   if(/^Linear layer/) {
      push @edges, sort {abs($a->[3])<=>abs($b->[3])} @layeredges;
      @layeredges = ();
      $layer++;
      $out = 0;
      next;
   }
   if(/^PReLU parameters: (.*)$/) {
      my $num = 0;
      push @prelus, map {[$layer, $num++, $_*1]} split(/\s+/, $1);
      next;
   }
   s/^\s*|\s*$//g;
   my @edgew = split(/\s+/, $_);
   my $bias = pop(@edgew)*1;
   pop(@edgew);
   @edgew = map {$_*1} @edgew;
   my $from = 0;
   @edgew = map {[$layer, $from++, $out, $_]} @edgew;
   push @edgew, [$layer, 'bias', $out, $bias];
   push @layeredges, @edgew;
   $out++;
}
push @edges, sort {abs($a->[3])<=>abs($b->[3])} @layeredges;
 
print encode_json({'edges'=>\@edges, 'prelus'=>\@prelus});
