#!/usr/bin/env bash

generalError()
{
    >&2 echo "clone_Inversion error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


while getopts ":M:i:o:" option
do
    case $option in
        M) master=$(readlink -f "${OPTARG}");;
        i) inputImage=$(readlink -f "${OPTARG}");;
        o) outputSino=$(readlink -f "${OPTARG}");;
        ?)
            >&2 echo "    Unknown option -${OPTARG}!"           
            exit 1;;
    esac
done
if [[ ! -e ${master} ]]; then >&2 echo "master ${master} does not exist!"; generalError "$0 $@" ; exit 1; fi
if [[ ! -e ${inputImage} ]]; then >&2 echo "inputImage ${inputImage} does not exist!"; generalError "$0 $@" ; exit 1; fi
if [[ ! -e $(dirname ${outputSino}) ]]; then >&2 echo "directory of outputSino ${outputSino} does not exist!"; generalError "$0 $@" ; exit 1; fi

executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[ ! -e ${plainParamsFile} ]]; then >&2 echo "plainParamsFile ${plainParamsFile} does not exist!"; generalError "$0 $@" ; exit 1; fi



echo "master = ${master}"
echo "inputImage = ${inputImage}"
echo "outputSino = ${outputSino}"
echo "plainParamsFile = ${plainParamsFile}"



master_new=$(./clone_Inversion.sh -M "${master}" -p _pre_ -s _suf_ -m shallow)
echo "master_new = ${master_new}"


bash "${plainParamsFile}" -a set -m "${master_new}" -F binaryFNames -f phantom -v "${inputImage}"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash "${plainParamsFile}" -a set -m "${master_new}" -F binaryFNames -f sino -v "${outputSino}"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash "${plainParamsFile}" -a set -m "${master_new}" -F reconParams -f isPhantomPresent -v 1
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash "${plainParamsFile}" -a set -m "${master_new}" -F reconParams -f isForwardProjectPhantom -v 1
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


./initialize.sh "${master_new}"


./purge_Inversion.sh -M "${master_new}" -m shallow




