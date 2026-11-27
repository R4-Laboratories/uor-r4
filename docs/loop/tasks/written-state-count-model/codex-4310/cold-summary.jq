def identity: [.id,.arm,.kind,.domain];
. as $matches |
if length < 14 or length != ($plan|length)
   or any(.[]; .passed!=true)
   or any(range(0;length); . as $i |
       ($matches[$i] | identity) != ($plan[$i] | identity)
       or $matches[$i].steps != ($plan[$i].steps|length))
then error("incomplete cold parity or unexpected replay ordering")
else {cases:length,words_covered_by_digests:(([.[].steps]|add)*4095),
      steps:([.[].steps]|add),groups:(group_by(.kind)|map({kind:.[0].kind,cases:length,steps:([.[].steps]|add)}))}
end
