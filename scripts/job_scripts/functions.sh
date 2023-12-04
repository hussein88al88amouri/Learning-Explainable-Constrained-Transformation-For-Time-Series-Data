#!/bin/bash


find_current_run() {
    local outdir=$1
    local label=$2
    local outfilebase="${outdir}/${label}"

    local i=1
    local outfile=${outfilebase}/run_${i}
    local highestrunoutfile=""
    while [ -f ${outfile}.id ]
    do
      	local highestrunoutfile=${outfile}
        local i=$[$i+1]
        local outfile=${outfilebase}/run_${i}
    done

    echo ${highestrunoutfile}
}


find_next_run() {
    local outdir=$1
    local label=$2

    local outfilebase="${outdir}/${label}"

    mkdir -p "${outfilebase}"

    local i=1
    local outfile=${outfilebase}/run_${i}
    while [ -f ${outfile}.id ]
    do
      	local i=$[$i+1]
        local outfile=${outfilebase}/run_${i}
    done

    echo ${outfile}
}
