def n(f): map(select(f))|length;
if ($device|length)!=($client|length) or ($device|length)==0 or
   any(range($device|length);
       ($device[.].arms.learned|del(.first_mismatch)) != ($client[.]|del(.first_mismatch))
       or $client[.].first_mismatch!=null)
then error("ordinary client differs from device learned path")
else {matching_rows:($client|length),terminal_rows:($client|n(.terminal)),
      transitions:([$client[].steps|length]|add)}
end
