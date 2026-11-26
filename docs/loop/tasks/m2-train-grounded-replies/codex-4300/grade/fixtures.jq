# Synthetic case for the report grader. Site 0 and site 1 have the same surface.
def unit($i): {start:(2*$i),end:(2*$i+1)};
def location($i): {bytes:unit($i),tokens:unit($i)};
def marker($i;$t): location($i)+{index:$i,text:$t,bpe_ids:[$i]};
def local($i;$transport;$joint):
    {site:$i,eligible:true,role:true,nonoverlap:true,strict_sign:true,cost:[0,$transport],
     joint:{best:[0,$joint,1],eligible:1,ties:1,rank:{cheaper:0,equal:1}}};
def trace($site;$cost):
    {links:[{word:0},{word:1}],tuple:{ordinal:0,eligible:1,ties:1,best:[0,$cost,1],rival:[null,null,null],sites:[$site,3,4]}};
def variant($g;$r;$jg;$jr;$winner):
    {request:0,row:0,sign:1,word:0,requested_site:0,rival_site:1,local_sites:
        [local(0;$g;$jg),local(1;$r;$jr),local(2;1;2)],
     requested:trace(0;$jg),rival:trace(1;$jr),native:trace($winner;2)};
{meta:[{row_index:0,sign:1,governor_word_index:0,requested_site_index:0,rival_site_index:1,
        governor:(marker(0;"V")|.bytes=unit(5)|.tokens=unit(5)),requested:marker(0;"A"),rival:marker(1;"A"),
        opposite:marker(3;"C"),heading:marker(4;"D")}],
 rows:[{text:"A A B C D V W",coordinate_map:{words:[location(5),location(6)],sites:[range(5)|location(.)]}}],
 before:[variant(5;8;12;9;1)],after:[variant(8;5;10;7;2)]}
