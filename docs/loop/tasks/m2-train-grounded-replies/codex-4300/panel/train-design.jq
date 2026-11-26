def require($condition; $reason):
    if $condition then . else error($reason) end;

def words:
    ascii_downcase | scan("[a-z]+");

($mixed[0] | map(.turns[1]) | map([words]) | add | unique) as $trained_words |
# The intervention adds only two selector lexemes. Both are full tokens in the
# signed vocabulary, and all the surrounding prose has a literal mixed-TRAIN witness.
require(all(["Ġfirst","Ġlater"][]; . as $word | $codec[0].model.vocab[$word] != null);
    "fixed selector intervention lacks a signed BPE lexeme") |
require(all($frames[0][]; . as $frame |
    [$frame.before, $frame.after, $frame.marks[]] |
    map(gsub("\\{[a-z_]+\\}";"") | [words]) | add | unique |
    all(.[]; . as $word | $trained_words | index($word) != null));
    "derived scope frame contains prose absent from mixed TRAIN") |
. as $grammar |
require((.wordings|length) == 4 and (.facts|length) == 2
    and ([.wordings[].family] == [$frames[0][].family]);
    "derived frames must align exactly to the four TRAIN families") |
require(all(.wordings[]; . as $wording |
    all($grammar.facts[];
        . as [$fact,$job,$home] |
        any($mixed[0][]; .family == $wording.family and .fact == $fact and
            any(.fact_spans[]; .role == "job" and .value == $job) and
            any(.fact_spans[]; .role == "home" and .value == $home))));
    "TRAIN source has no literal witness for a family/fact configuration") |
.wordings |= map(.family as $name |
    . + {contrast:($frames[0][] | select(.family == $name) | del(.family))})
