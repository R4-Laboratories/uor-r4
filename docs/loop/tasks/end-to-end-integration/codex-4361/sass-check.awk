# Audit the instruction roots of the sm_89 machine code actually used in this run.
# Input is the complete stdout of cuobjdump --dump-sass for one cubin. FLO is the
# integer find-leading-one instruction. Reject unknown roots and empty inputs.
BEGIN {
    split("BAR BRA BREAK BSSY BSYNC CALL CS2R EXIT FLO IADD3 IMAD IMNMX ISETP LDG LDL LDS LEA LOP3 MOV NOP P2R PLOP3 PRMT R2UR RED S2R S2UR SEL SHF STG STL STS UIADD3 UIMAD UISETP ULDC ULEA ULOP3 UMOV USEL USHF WARPSYNC YIELD", list)
    for (i in list) allowed[list[i]] = 1
}
/^[[:space:]]*\/\*[0-9A-Fa-f]+\*\// {
    field = 2
    if ($field ~ /^@/) field++
    mnemonic = $field
    sub(/[.;].*$/, "", mnemonic)
    instructions++
    if (!(mnemonic in allowed)) {
        print "rejected", $field, "at", $1
        rejected++
    }
}
END {
    print "instructions", instructions+0, "rejected", rejected+0
    if (!instructions || rejected) exit 1
}
