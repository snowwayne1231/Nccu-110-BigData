??/
?&?&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??.
?
Embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*%
shared_nameEmbedding/embeddings

(Embedding/embeddings/Read/ReadVariableOpReadVariableOpEmbedding/embeddings* 
_output_shapes
:
??*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1029*
value_dtype0	
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameAdam/Embedding/embeddings/m
?
/Adam/Embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/Embedding/embeddings/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/Embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameAdam/Embedding/embeddings/v
?
/Adam/Embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/Embedding/embeddings/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes

:??*
dtype0*??
value??B????BtheBaBofBtoBandBiBinBbitcoinBthatBonBforBhaBitBwithBaloB201dBcryptoBbyBhaveBcryptocurrencyBareBread:BfromBthiBnewBanBbeBbeenBwillBexchangeBatBwaBdigitalBmarketBbankBtheirBoverBcompanyBwhichBheBmoreBlatBaboutBaetBoneBpriceBcahBnowBnotBorBtradingB	financialBcanBbutBminingB	accordingBhowBtheyBhiBafterB	announcedBmillionBu..B
blockchainBallBbtcBthanBcryptocurrencieBintoBerviceBomeBthereBotherBplatformB2019BfirtBlikeBtokenBpeopleBinceByearBupBduringBuerBweekBwhoByouBcoinBmanyBreportBnumberBcurrencyBtwoB
governmentBtimeBfundBmotBwereBwhenBnetworkBbchBfirmBpaymentBcentralBcalledBjutBbillionBwalletBtheeBwhileBprojectB2013BworldB	invetmentBoutB
tranactionBfewBinvetorBcountryBweBueBmoneyB00a0BglobalBrecentlyBaidBrecentBhowever,BdayBhadBaroundBmayBtateBuingBuchBayBvalueBevenBpublihedBplanBwouldBlotBwhatBperBreportedBonlyBtopBtradeBagaintBoBcouldBbeingBifBnoBwellBdataBouthBmonthBethereumBhighBbitcoin.BfutureBcompanieBlargetBcourtBbetweenBpatBrevealedBpopularBlaunchBminerB	developerBgroupBtaxBorderByear.BthreeBmakeBofferingBaccountBecuritieBallowBcutomerB	includingBdecentralizedB	followingByourBthemBceoBthroughBbuineBtillBofferBuedB	communityBmediaBbelieveBworthBblockBmadeBdownBpublicB
regulationBlawBbitcoin,BanotherBvolumeBlaunchedB	regulatorBlocalBbackBbitcoin.comBtakeBourBeconomicBendBupportBchinaBunderByear,BmajorBanyBwayB
regulatoryBteamBcommiionBprotocolBknownBfinanceBcomeBuBdevelopmentBprovideB
technologyBlatetBytemBtraderBbecomeBtodayBindutryBgetBwhereBdollarBownBwantBjapanBeenBclaimBnationalBbeforeBwithinB	currencieBnftBetBdoBweek,BcoinbaeBamericanB(btc)BacroBpartBatohiBinteretBbillBofficialBhelpBveryB	exchange,BtartedBcryptocurrencie.BfederalBleadingBreerveBchineeBwebiteBruianBbothBnextBvirtualBbecaueB201ctheB	operationBmuchBpreidentB	currentlyBcaeBtockBbankingB
reportedlyBiuedBpotBnew.bitcoin.comBproductBeeBonlineB
individualBpaceB2014B(bch)BcontinueBweek.BinitialBcurrentBlegalB	exchange.BfiatBcardBeveryBopenBviaBneedBdueBapplicationBindiaBiueBgoldBpayBthenBicoBtwitterBecondBcapitalBocialB	explainedBcreatedBhouldBreleaedBecurityBeveralBecBalreadyBpercentBcoreBthoeBincludeB	differentBindianBthinkBbuyBwithoutBdepiteBeconomyBtotalBpowerBfounderBprivateBmemberB
ignificantBmarchBinternationalBgreatBday.BcountrieBpurchaeBclientBinformationBbaedBaleBfeatureBcontractBannouncementBameBtoldBmarket.BamongBrateBchangeBacceptBgrowingBfeeBcreateBoldBappBhareBacceBjapaneeBreearchBlargeBfourBmoveBlowBlongBworkingBrightBgainBaugutBtoreBplaceBtartBfiveBagencyBbigBmeanBeuropeanBlookBcryptocurrency.BaddedBtatementB	tablecoinBmakingBfiledBbinanceBincreaeBkoreanBtowardBlitBjuneBbuineeBgameBeachBkoreaBgoingBearlyB
intitutionBamountBhardBformerBproviderBmarket,BunitedBoftwareBtetherBactionBchainBaet.BbehindBcryptocurrencie,BleBheldBforkBaimBaprilBorganizationBixBexplainBtakingBmonth,BbanB
authoritieBfoundB	platform.BgoodB2019reBknowBmonth.BjulyB10BoptionBtime,BruleBall-timeBadoptionBtime.BdecemberBbetBworkBtermBholdingBdetailBolutionBgiveBfarBcallBbank,BhardwareB
managementBjanuaryBableB	availableBrecordBkeyBellBdemandBrikBtartupBdeciionBonceBmightBdailyBreport,B	executiveBinternetBtookBfindBday,BudBeventBruiaBeemBholdBformBchiefBincreaedBoffBeptemberBintitutionalBdefiBvariouBrealBwebBthingBcryptocurrency,BledgerBhahrateBconfirmB	authorityB2018BwhyBawBpreBcityBtoolBroughlyB	interviewBdealBmartB	regardingBervice.B
departmentBaddBhavingBvideoBeverBapproximatelyBnovemberBfollowBtranaction.BtraditionalB24BcreatorByorkBiteBmonetaryBalmotBwrightBlookingBwhetherBoctoberBofficeBentireBtudyBtakenBecoytemBhotBraiedB
launderingBalternativeB	acceptingBprivacyBdroppedBupremeBregionBatmBfullBcountry.BrunB
partnerhipBquiteBfreeBgrowthBtrutBcitizenBrevenueBmaiveBenforcementBrewardBthirdBreleaeBquetionB
meanwhile,B1Bnetwork.Bwell.BfundingBerieBmerchantBanalytBhoueBgiantBfurtherBmondayB
introducedBideaBgoB2020,BoonBdontBourceBminiterBpoliceBnon-fungibleBmarketplaceBleatBellingB	launchingBhongBcloeB	operatingBtrillionBcamBpointBmutBethBaboveBpoolBaddingBproceB
conferenceBpropoedBuntilBpaperBentBdocumentBbiggetBlawuitBdoeBdecidedB	proponentBnameBkeepBhedgeB201caB2017BagainBpoitionBmallBanyoneBleadBetfBentranceB
derivativeBmobileBaverageB	allegedlyBlitedBexpectedBdicuedBcapitalizationBrevealBpropoalBhourBloeBafricanBherBwednedayBgettingBworld.BverionBunitBfebruaryBcontrolBakedBtuedayBnow,BtrendBeffortBchairmanB	platform,BmultipleBharedBarticleBabilityBattackB2020B
00a0yahoo!BreaonBfridayB100BtypeBrieBcriminalB(ec)BurveyBchemeBaddreBoftenBcompany,BcomingBinvetBbitmainB2021,B00a0bitcoinB
throughoutBputBimilarBheadBrelatedBnearlyBhalfBminitryBnakamotoBapprovedBrippleBriingBmanagerBmachineBgotBeekBdidBthurdayBupcomingBpeer-to-peerBletB	committeeB20BnoteBcotBremainBbetterBwarningBventureBappearBago,B	valuationBownerBalongideBtradedBreceivedBpoibleBpolicyBnorthBnationBfintechB30BubjectBleverageBenergyBdropB12BlittleBinvetor.Bindutry.Baet,BreachedB	potentialBcameBadditionBnodeBledBhigherBbringBagoBregulateBcertainBthouandB	maintreamBdogecoinBcenterBleaderBmeetingBmarkBinvetigationB50BingleBcommentBretailB	providingBplayerBoperatorB	aociationBhackerBoperateBneverByetBtreetBtartingBliveBwallet.BcreditBconcernBallegedBprogramBdeignedBcoverBervice,BenableBearlierBdarknetBcapB	venezuelaBinvetingB	importantBhitB	currency.BtechBgivenBdetailedBdeviceBbillion.BrigBpublicationB	moreover,BeekingBeditionBcreatingB2018,BtranferBtatedBlpBinfratructureBcoureBbuyingBupdateBtoken.BpaidBboardBmineBhimBdepoitBawayBlitingBtelaBmajorityBcoin.BbullBupgradeBtweetedB
coronaviruBgeneralBbtc,Bytem.BletterBit.BaltcoinBreceiveBoverallBcodeBrepreentativeBmr.BforumBenthuiatBbenefitBactBtryingBtooBprofitB	frameworkBtoday,B	regulatedB201cweBfund.BexpectB2020.BlevelB201ciBmatterB	bitcoinerBaddreeB	worldwideBupporterBrangeBinvolvedB
electronicB
ditributedB	reported.Bfuture.BdirectorBtepBtenBtreauryBfactBectorB(nft)B00a0theBwallBmainBchargeBmodelBeeingB	reearcherBproblemBconceptBviewBmukBlikelyBintance,BforceB&BreidentBmonday,BcrimeBreultBmethodBetherBdraftBdoentBofficerBconumerBblogBoutletBdicuionBbuiltB2019,Btoday.B	recently,BpreviouBholderBattemptBpartnerBmeaureBectionBwrittenBloanBblockchain.BtandardBpoitiveBmillion.BmanBvarietyBenterBbelowBwentB$1BreallyBgoogleBfollowedBexchange-tradedBeffectB	activitieBlitecoinBupplyBuniqueBpayment.BmanufacturerBareaBafricaB2BjobBbackedBnearBportalBimpleBhomeBbitpayB	conductedBreturnB	previoulyBexpertBcah.BaociatedBplanningBfoundedB	etablihedBelonB	developedB201cbitcoinBperiodBkongBfedB2018.BtelegramBparticipantBhortBfilingBforeignBexitingBignificantlyBeayBcah,BanalyiB2017.B2017,BwiBomethingBfigureBditrictBaccuedBwallet,B
venezuelanBtechnology.BraieBgainedBbtc.BbeganB	liquidityBbitfinexBgraycaleBelectricityB
currencie.B	confirmedBtweetBrequetBreponeBleftBhopeBgovernorBliceneBincomeBfaceBeconomitB
withdrawalBnotedBfindingBeveryoneBdicuingBthoughBproceedBlowerB
concerningBhitoryBregulation.BpurchaedB	politicalBlawmakerB	inflationBhandBuggetBroundBquarterB	currency,Bcovid-19B	autralianBalwayB15BhutBamericaBallowingB201d.Bworld,BpecificB	dedicatedB	wedneday,BtetBillegalBgroup,BgoxBfocuB
co-founderBarretedB	lightningBjumpedB	introduceBindicateBhundredB	extremelyBbullihBperonBofferedB	increaingBhereBcomputerBvalue.BbeginBrepreentBcauedBproject.B
officiallyBlauncheB
innovationBinc.Bhour.BvictimBtueday,BroleB7B2021B	univerityBtrongB	regiteredBminedBexpandBcrypto.B5B2021.BtalkBprovidedBpendBcoin,BcanadianBwhoeBtimuluBtechnologieBnegativeBearnBdebateBbuildBbearBentrepreneurBboughtB2019.BtalkedBmanagedBhackB
experienceBdonationBxrpBeconomy.BauctionBphyicalBmovingBmoney.B	continuedBcalingB	attentionBactivityBthem.BregitrationBnew,BignedBgrownBfileBconenuBrunningBemployeeBcraigBbuildingBbitBannualBthoughtBalongBtouchedBlackBimplementationBaccount.B1,BwhiteBuer.BpendingBfriday,B
foundationBecureBdealingBblockchain,BbecomingBaid:BrequireBnoticeBendingBbritihBtrackBprotectBmyBfanBeizedBdollar.BconiderBclaimedBreachBpotedBjudgeBizeBnetwork,BgivingBwapBtemmingBtate.BquotedBkindBcoupleBmorningBfund,BelBtrumpBperonalBpace.BlocatedB
currently,Bbank.BauthorB3B
well-knownBfacebookBanonymouBturnBregionalBjohnBfirm,BcutodyBadditionally,BtatiticBirBfocuedBcrypto-relatedBcreationBwealthBukBreport.BproceingBpecialBlightBcomparedBtetingBjoinBimplyBgamingB
legilationBhumanBfraudBfact,BdoingBcrypto,BactiveB201d,BtoryBmeetBlabBjuticeBenoughBdoubleBcommonBcnbcBcannotB40B(ico)BnigeriaBlimitedB	involvingBartBaicBtopicBreachingB	portfolioBiranBintelligenceBindexBchangedBbriefBallowedBwholeBthurday,BpartyBnativeBgrowBfullyBcontentBwarnedBulbrichtBroadBreviewBlaterB	corporateBapprovalB	agreementB
interetingB	ethereum,Btoken,Bthen,BplayBoriginalBlongerB	ettlementBetateBkleimanB
developingBcmeBtrategyBnewlyBmore.Binvetor,B
initiativeBaid.BloBitelfB	conideredB2026BratherBopinionBmeageBdicuB
additionalB(defi)Becoytem.BdirectlyBconductBbillionaireBvolume.BongoingB	interetedBignBgoldmanB	challengeBcentralizedBago.BabcBwriting,BurgeBmovedBlaw.Bytem,BredditBpropertyBproject,B
profeionalBnumerouBbookBakingB	technicalBmcafeeBhoveringBthreatB	preparingBlifeBwordBprice.BmonthlyBgreaterBdirectB
difficultyBcorporationBcitingB	bloombergB	reportingB
politicianBpaypalB	epeciallyB	crackdownBcountry,BcheckBtranaction,Bthurday.BphoneBmichaelB
everythingBactuallyBtrading.B
operation.BlineBeuropeBcampaignB	undertandBrequirementBpayingB	partneredBit,B
commercialBchina,BalthoughB(eth)BtorageBintBhearingBeuroBeuBdevelopBdeputyBunionBubmittedBpriorBnamedBhappenB	guidelineBemailBdidntBcriiB	wedneday.BubidiaryBpageBmonday.BdavidBchain.B4BlocationBlendingBerveB
community.B	beginningBattorneyB21BwroteBreadyBpentBpairBmorganBilkBfamilyBdebitB
conideringBclaimingBamidBvolume,BtolenBpubliclyBmoney,BexitBemergingB	dominanceBcircleBbalanceBwhaleBurpaedBupdatedBpokeBmovementBjan.B
invetment.BfacingBenatorBdecribedB
complianceBcapital,B11B
00a0indianBprimeBpikeBpaedBmemeBhowingBexpoureBclaBbecameBinteadBeitherBearchBdeBchannelBartitB8B
witzerlandB
protectionBpieceBpayment,BfarmBchoolBweveBtrading,BthankBtakBpatentBlimitBinvetedBdoneBbch,ButilizeBincreainglyBexperiencedB
revolutionBpreadB
leveragingB	countrie.BaheadB	00a0japanBrequiredBrecordedBponziBperformanceBonchainBmetricBforwardBbugBunveiledB
parliamentBeightBtravelB
regulator,BproducedBopportunityBneededBloingBlewBfinallyB
facilitateB	commodityB$100BudtBtueday.BomeoneBpromieBhuobiBacquiredB201ctoB201citBzoneBtechnology,BtatuBrepublicBlockedBlicenedBgermanBfallBfakeBcompany.BaiaB
remittanceBobtainBincludedBearningBcoldB
anti-moneyB	protocol.Bindutry,BhighetBhandleB	expandingBetimateBchargedBbaicBautraliaB25BuponBtaxeBregulation,B
particularBimpactBegwit2xBdrugB
collectionBbetaB18Buer,BmoneroBmeantBlateBjoinedBetimatedBcutBclearBbrandB	advantageB
acquiitionBupportedB
predictionBp2pBmomentBimpoedBcheduledBapproachBannounceByoutubeBwebite.B	poibilityBpeterBgeneiBfurthermore,BforcedBdeepBdateBagencieBafeBu.k.BpeakingBonecoinBgoalBfriday.BanwerBadB6Btate,BownedBenateBchanceBbBu..,BtiedBtarBretBnov.BinternalB
innovativeBingaporeBbringingBannouncement,Bproduct.B
popularityBpikedBlawyerBgoeB	financingBfatBfailedBevidenceBbearihBvatBtudentBtargetBnotingBhotedBhighlyBfearB	concernedBcompleteBadvocateByuanB	upportingBuccefulBtorieBpromoteBproduceBplitBcontroverialB90BroBpecificallyBlondonBgaveBexampleBanctionB
urroundingB
retrictionBregiterBpuhBmedia.BheadlineBgainingBcommunicationBbillion,BamBaimedBwarBthemelveBmining,BentitieBecretaryBcollectibleBavingBagreedB	addition,Bupdate:BquicklyB	proecutorBleavingBlearnBiuingB	implementB	highlightBgovernment.BdexBdaily,BcantBbidenBachB13BpreventBelectionBector.B
commiionerBchipBcarBbotBappleBurgedB	purchaingB
pertainingBpace,Bnow.BmtBlocalbitcoinBiranianBinvolveBheardBclient.Bwebite,BundayBpublication,BpaulBnothingBhownBfamouBdozenBcouncilB17B1,000Bunday,BreourceBnetBmakerBfinalBbannedB60B00a0aBvalue,BtogetherBreuterB	predictedBpreciouBillicitBgrantedBeniorBdecribeBanalyticB15,BwontB
volatilityBviceB	reponibleB
programmerBprice,BorderedBfidelityBenureBbuine.BayingBattractBadminitrationBweekendBtrueBregardBmaBintituteBimproveB
currencie,Bbinance,Baid,BaffectedB
worldwide.BtemBproceorBpreidentialBmining.BhalvingBfurther,Bdec.BacceptedBwideB	ranomwareBraiingBoct.BnigerianBmetalBmaduroBindependentB12,ByoungBwappingBquareBprionBopportunitieBfuture,BenteredBcaughtBcallingBbelow.BagentB80Bwrote:BtenderB
regulatingBreadBpreentB
percentageBoilBfollowerBcutomer.BukraineBpurpoeBmiletoneB	long-termB	knowledgeBimmediatelyBi,Bfeb.BexpanionBcboeBcae,BbuyerB(btc),ButilityBtoppedBolution.BmyriadB
generationBexpreedBdometicBapproveB24-hourBzeroBtaxationBtatingBpetitionBlater,B
integratedBimfBhackedB
explainingBenvironmentBeailyBcaueBadoptedBacquireBtrade.BproofBmicrotrategyBlargerBgaBfreedomBeyeBbondBandroidBalo,Baccount,Bway.BwaveBthat,BtayBregion.BrbiBrate.BparticularlyBilandBhearing,BhearB
fraudulentBdata.Bd83dBchairBagain.BwaitingBpolicieB	marketingBhour,BgermanyBfee.BequityBdeclineBcyberB	complaintB14ByieldBjpmorganBhugeBetablihBdebtB	countrie,BcompetitionBchartB	candidateBcanadaB23B19B16BwonBpetroBotherwieBo-calledBlaw,BhappenedBdollar,BdeclaredBcoveredBcloedB	chainalyiB	brokerageBbankerBbaeBwidelyBverificationBureBummerBtellB
philippineBmetBmanageBkrakenBituationBidentityB
crypto-aetB	companie.BcaptureBapplication.BalvadorBaianBwithdrawBthailandBteadilyBtanceB	receivingB	prominentBopeningBminer.Bminer,B
invetment,B	eentiallyB	companie,BbodyBbegunBapplyB	amendmentB	ukrainianBplannedBmillion,BintegrationB	initiallyBgold,BgmoBexahahBentimentBdata,BcutomBcam.BbrokerBbootBbolterBbadBappliedB3,Bthi,BprobablyBopenedB
monitoringBmikeBjoiningBjapan,BinfamouBimplementedBhelpedBengineerBeionB	cofounderBcard.BbreakBavoidB	algorithmBakB9BuuallyBpracticeBpowellBmillion)BlayerBjameBinuranceBeconomy,BeatB	completedBcivilBbureauBbroughtBblockchain-baedBantminerBale.B
activitie.B2018bitcoinBtated:BeriouBanythingB70B11,BvaluedBtipBruia,B	protocol,BobtainedBnadaqBindia,BhelpingBfrenchBfavoriteB
community,Bcae.B51B2016BurveillanceBtandBnatureBmanagement,BjumpBideBhantBhahBgraphicBfightBfieldBeth,BentityBapp,B$10BwinBratingBrankingB	provider,Binc.,BfourthBentitledBdnmBdmmBcontainBcommandB	coincheckBcardanoBbch.BbakktBupendedBturnedB	offering.BlivingBiraeliB	intentionBelectronBelectricB	dicoveredB10,000BportionBoutideBnineBlibraBimprovementBhopBexample,BegwitBdonaldBcaleBbunchB500B201cinBreliefBprofileBpreentedBpilotBjuridictionB	initiatedBhigh.BgreenBexitenceB	difficultBchooeBarretBweeklyBrecoveryBproce.BportBoughtBmarginBmaintainB	leveragedBharingBaturday,B200ByeterdayBwelcomeBtriedBtheftBtabilityBrankBpartieBoperatedB	martphoneBluxuryBimageBgatheredBg20BfavorBcongreBchina.Bcheme.BtouchB	repondentBpandemicBpainBotcBmt.BmonitorBitemBintendBgiftBfamiliarBdetail.Bcoinbae,B	automatedBaffectB201cthiBvoteBvBtargetedBtableBrealityBpremiumBpowerfulBpanihBnotoriouBmaltaBlibertarianBgoneBdroppingB
differenceBcloingBcbnBufferedBtrialBtonB	robinhoodBredBprofeorBextenionBditributionBcrahB
connectionBcollaborationBcammerBbitcoin-baedB
00a0marketBwritingBunit.BtokyoBtalkingBreduceBreaderBpeakBdicueBcomplyBclubBceaeBcanaanBaturdayBadoptBwell,BucceBrichBrelationhipBprovinceBoundBocietyB	mentionedBintitution.BintalledBeurope,BeparateBenteringBdevelopment.BconverationBceo,BaveBactorBwetBv.B	utilizingB	uccefullyB	televiionB	recognizeBpredictBparticipateBometimeBhypeBfee,BeaierBdeathBcryptocurrency-relatedBblackBbelievedB00a0outhByenBvaluableBtryBthaiB	promotingBproduct,BprepareBlinkBleaveBlatinBlanguageBlandBjointBignatureBgalaxyBerc20Battack.BappealB
acceptanceB$2BwantedB	upervioryBunknownBproveBpromiedBlegacyBintroductionBhubBgovernment,BfranceBfeaturedB	facilitieBdeclareB
conumptionB	contract.BcombinedBcheapBcbdcBaffairBwarrenBviionBummitBrumorBpoweredBpoeBovereaBmediumBmedia,BkycBintroducingBgold.Bfork.Bevent,B
equivalentBdubbedB	determineBdeignBdecadeBcauingBbidBappearedB5,B17,BtezoBtexaBrulingBprettyBother.BkeepingBfallingBfailBfacilityB
explained:B	enterprieBdiviionBconnectB
californiaBbitmexBargumentBtageBpot,BpodcatBouthernBmanagingBkorea,BjoeBignalBhotingBfriendB	ethereum.Belaborated:BdarkBcriticalBcloudBchicagoBbithumbBargueB14,B(etf)B	targetingB
tablecoin,BrienBquickBpoiblyBnoticedBmiamiBmallerBloveBiuanceBinventorBinightB	happeningBgain.BfineBfilledBecoytem,BdipBclaicBchangingBbuinee.Bblock.BactivelyB8,B7,B20,B(udt)ByouveB
relativelyBpuhingBnoncutodialBfaterBexpectationBcloerBbrowerBageB6,B25,B2010B10,BvalleyBruledB	peculatorBfoodBeoBcenorhipBcard,Bapp.B31,B23,BuppoedBtitledBretailerBreitanceBprotetBpollBpaingB	novogratzBmeagingBmeBinideBflahBexpandedBettingBeffectivelyBdr.BdogeBchriBcannabiBbrianB	bitfinex,B
announcingB35B201cdigitalB2019mB00a0howB	unlimitedBrogerBpropoeBplacedBpictureB	intrumentBguidanceBfirm.BfactorBexcitedBentencedBelectedBdicloedB	clarifiedB
201ccryptoB(btc).BuageBprogreBout.BofflineBmovieBmotlyBmiionBlidBlegalizeBinvetigatingBhiftBfork,BfeelB	equipmentB	effectiveBeeminglyBdriveBdeniedBbvBadvioryB	adoption.B(bch),BwhoppingBufferingBturningBreferredB	referenceBrapidlyBownerhipBoon.BmocowBgame,BclimbedBcapturedBbreakingBbanningB	attractedB4,B28,BwortB	wahingtonBundertandingBtwitter,BtringBremovedB
recognizedB	organizedBoption.BioBguideBfinance,BfailingBexcluiveBeaingB	comparionBchapterBchange.B
attemptingB	aggregateB19,B1.B
regulator.BoutbreakBmuicB	matercardBiuerBico,BhimelfBftxBflowB	ecuritie.BecretBdoubledBcutomer,BcompreheniveBcienceB300B$4BzcahBtronB
tranferredBtoredBtock,B	tatement,Brik.BpromiingB	plaintiffBon-chainBnecearyBlinkedBincidentBhapehiftBforkedBentertainmentBemergedB
definitionB	contract,BbiBbeyondBapectB22B201conBwork.BuggetedBtrader.BtrackingBreporterBremarkB
profitableBprizeBprimaryBpeople.BparticipatingBpackageBmanipulationBliquidBiniderBheadquarteredB	generatedB	exploringBentirelyBeaonBcoure,BconvertBcomprieBchoiceBchiffBadvancedBachieveB13,B(eth),B(bch).Bventure,Bud.BtaxpayerB
tablecoin.B	nonprofitBlargelyBipoB	indicatedB
identifiedBguiltyBgroup.BgithubBfunctionBexploreBeedBdomainB	directiveB
developer,BcrackBbettingBarmBafrica,B(ath)BwrongBtrategitBreplaceBremoveBproducerBpreureBpotentiallyBplan.Bperiod.BparkBparentBindia.BhortageBhigh,BgloballyBfar,BenglandBdeliverB	competingB48B22,B2014,BwinnerBway,BwarnBvolatileBupwardBu,BtrategicBticketBthem,BtetheredBrallyB
populationBphaeBpeedBoverightBmaliciouBleeBjackBhitory.BgiveawayBgenerateBdcafBcftcB	brazilianBbittrexBbill,B30,B2,B(ico).BwitneBwaitBupectedBtechnologicalBrobertBregularBrate,Bpercent.BolanaBnadaq-litedB
governanceBevent.Benvironment.BenablingB	conditionBbtc/udBbattleBawareBadviorB201cifB2016,B(cbdc)BunlikeBtill,BpuhedB
peculationBmotionBkeptBheightB	emergencyBdown.Bdaily:Bcrypto-friendlyB	cahhuffleBarentBamazonB	201cthereB2013.B(nadaq:BupendBteam.BtankB
productionBpeechBpaBlockdownB	gatheringBfraud.BfrancicoB
eventuallyBethereum-baedBembraceBdaveBaction.B21,B$30BroundupBpumpBpointedBplace.BnorBmindBlaBiraelBinformedB	indutrialBhedBgatherBfractionBfellBdicloeBdecreeBdahBcrii.B	characterBcareBcam,BcainoBbridgeBbeijingBale,Bagain,B28B24,B18,B$3BwillingB	widepreadBvaneckBunregiteredBtate:BrepreentingBputinBpoorB
operation,BmultinationalBmortgageBlatterBheatBforexBecond-largetBdevelopment,BbudgetBbranchBbeideBanticipatedBadviceB45BwitneedBvehicleBurgingBuncertaintyBuefulBtouchingBtomBtickerBover-the-counterBoutlookBnotableBiue.BintendedBinitB
influencerBimpoeBico.BgroundBfrehBetablihmentB	detailingBdefeneBdappB
continuingB	connectedBconiderableB
completelyBcitieB	capturingBbubbleB
allegationBadding:Badded.BvitalikBtreedBtheyveBrepondedBpreadingBpeakerBone.Bjune,BjimBintantB	inceptionBhotelBhand,BgrantBformedBfar.BervedB	efficientBcubanBcomplexBcity,Bcapitalization.Battack,BandrewB2015B$1,000BwatchdogBupport.BuggetingBu.BturkihBtroubleBtrezorBround.BremainedBrelevantBrejectedBplayingBpickBperformBortB
originallyBmarketplace.BjeromeBholding.BfortuneBfloridaBfeature.BexploitBexamineBept.BdippedBadultBacceibleB27B16,B150B100,000B(lp)BzimbabweBunprecedentedBthuB	revealingB	real-timeBprofit.BproBoccurredBnft.BmintedBmintBlevel.B
importanceBfocuingBfightingBfairB
explained.BelectBdownloadBdoorBdeutcheBdecentBcarriedBcapital.BbrokeB27,B$20BworkerBup.BunableBtrut,Btrade,Bthough,Btechnologie,BtaffBrecommendationBreaon.BrapidBquoteBnchainBgpuBettleBdominantBcyberecurityB
criticizedBbill.BbarBaugut,B	argentinaB
advertiingBactualBactingBa.m.B201ccryptocurrencyBwinterBwatchingBurveyedBtownB	tokenizedB	territoryBtechnologie.Bteam,B	provider.BpreparedBpoloniexBparticipationBparkedBontoBokexBo,BminuteBmiingBmentionBmay,BkimB	incentiveBhibaBexperiencingBetf,B	educationB	december,BcircularBcentreB	celebrateBbtc-eBbroadcatBartworkBanywhereB	alexanderB00e9B$1.5B
worldwide,Btrut.BtahBreerve.B	publihingBpleaedBplantBperfectBnorthernBmodernBmodel.Bloe.BlatelyB	journalitBjeffBinvolvementB	inpectionBindutrieBhahpowerBfahion.B
experimentBerverBeducationalBcutodialB
cro-borderBcirculatingBchain,Bbuy,BattackerB29,B$300Bxrp,BteveBtaxe.BroeBrevolution:BreflectBprogram.Bp.m.BoutlinedBmanager,BlnB
interview,B	interfaceBintellectualB	immediateBhuttingBhowedBhaventBhare.BhackingBfallenBepiodeB	encryptedBeion.BecbBdevice.Bdecade.BdaiBcryptoconomyBcriticBcooperationBconfirmationBbittampBbaicallyBattendeeBapplication,BamungBadmittedB2014.B00a0160B(ec),B(doj)Bwriting.BvendorBtrutedB	trugglingBtrader,BtippingBtellerBpurpoe.BparityBnairaBmechanimBmatter.BmainnetBhow.BheavyBheavilyBfraud,Bfinance.BentenceB	encourageB	emphaizedBecuredBdrivenBdimonBdeemedB	decribingB	committedBcombatB	collectedBcirculationB
authorizedB	appointedBand,B	acquiringB9,B32B29B26,B201cwillB201canB(pboc)BymbolBtwitter.BthinkingBtappedBrule.BreliableBputtingBprintingBpotponeB
performingBokex,BnowdenB	nakamoto,BmiddleBlearningBjpBjapan.Bjanuary,BjamieBhealthBfrancico-baedBexactlyBeurope.BemiconductorBelaborating:BeaternBdreamBcountB	continentBcollectBchoenBcahfuionBbrazilB
anniveraryBallianceB201cblockchainBworkedBreviedBregimeBpurchae.BpringBpeople,BpanelB	optimiticB	offering,BlegallyBlaundering.BkingB	influenceBi.BeverydayBetf.B	editorialBdoubtBdonatedB	document,B	ditributeB	deliveredBdecideBcottB
controveryB	computingB
commitmentBcharleBcapacityBcanada,BbuttonB26B(dex)ByoullBwedihB
venezuela,B
underlyingB
ultimatelyBuitBtreBtranparencyBtanleyBretrictB	retaurantBquantumBproof-of-workBpatternBparticipatedBovereignB
openbazaarBoldetBnewpaperBnbaBnaturalB	nakamoto.BmtiBmember,Bmarketplace,BmalwareBkorea.BhopingBgaryBgame.BfrontBfootballBfifthBenvironmentalBelf-proclaimedB
eentially,BcutodianBcompeteBcommerceBcodebaeBcloelyBchaeBcfdBbuterinBbuine,BbrotherBbehalfBalphaB00a0andBwakeBviitorBverBvenueBupenionBupectB	typicallyBtvBturkeyB	trademarkBrkBrelatingBregion,Bolution,B	november,Bnew.Bmarch,BmagazineB	litecoin,BkucoinBkenyaBkarpeleBitalianB
invetigateBinuBintallBha256BfundamentalBfunctionalityBfbiBether,B	egregatedBearcheB
e-commerceBdirectedBdicountBdeadlineBbreachB	bitliceneB
attributedBagency,BadvanceB2015,B15.B$500BwatchB	violationBurprieBumBu..-baedBtieBroomBrequetedBrepondB	regularlyBrankedBq1Bpublication.BpotingBpmB
peculativeBmomentumBminimumBltd.BkingdomBimplementingBglobeB
fundraiingB	february,BextendedBextendB
electricalBecurity,Bector,BeconomieBdutchBdrivingBdoreyBdemand.BdeitB
commoditieBclaim.BblockfiBbitcoin?BaudienceBaociateBairB
accountingB2013,B	00a0ruianB00a0newB$50B$200BwuBtightenBtevenBtealingBryanBruia.BpropectBpromptedB	preident,BpluBpartner,Bone,Bmodel,BmixingBmidtBmicrooftBmenBlowlyBlowetBlenderBintitution,B	integrateBidentifyBhortlyBhoppingBfocueBfamedBengagedBeizureB
determinedBdate.B
cypherpunkB
blocktreamBbithumb,B	autonomouBathBapril,BandreaBamerica,BallegeBagreeB201cnotB2016.BzealandBweternBuppoedlyBunnamedBunday.Bud,B	tranitionBtore.BtheoryBt.BreultedBreearch,BrareBpolkadotB	pokeperonBpoition.Bpetro,Borganization,BoftBmoment,BmanufacturingBltcB
legilativeBjuly,Bintead,BholidayBexponentiallyBervingBeraBdumpB	dependingBdeclarationBbuinee,BbuffettB
bitconnectBbiggerB
bankruptcyB	attemptedBamerica.BalertBaddreedB(cftc)B$10,000Bweekend,BuniwapBuniveralBunB
ubequentlyBtructureBtrafficBthird-partyBtatBtakedBradioBquarter.Bpublic.B	producingBprivacy.B	primarilyB
portfolio.Bmanagement.Blondon-baedBlately,Blab,BinvokedB	indicatorBhighlightedBhiddenBhandfulBgrewBgapBfcaBfatfBextremeBexplorerB	eptember,BenabledBdeployedBdeireBdefineBdecentralizationBdecade,BcroedBcourt.B
conductingBcatBbelaruBaying:BawareneBarea.Badded:B	activity.B(rbi)B(p2p)BworriedBwomenBveteranBtoneBteadyBreultingBranBquotingBproviionBprogrammingBprivacy-centricBorder.BoaredBnvidiaBnote.BnicolaBneverthele,Bnation.BmintingBltc,BkepticBinfluentialBimplicationBgbtcBfixedBettledBdeciion.Bdeal.BcontratB	compriingBcollapeBco.Bclient,BcitedBbrancheB	averagingBalo:Ball,Baddre.B7nmBwormholeBwikiBtruggleBtrictBtore,BthreadBthat:BreferBqrBproceeBproblem.Bpoint,Bplan,BperhapBoption,Boctober,Bmay.BkillBheadedB
hareholderBfilmBfaleB	evolutionBend.B
emphaizingBdeadBcreditorBcreativeBcrackingBcoveringBchannel.BceneB
celebritieBbukeleBborderB
attractingB	anonymityBactivitBacademicB2.0B$6B$5B$400Byet.Bupdate,BudcBtudy,BtruteeBterroritBreadingBpublihB	propertieBprivacy,B
perpectiveBpbocBpawordBon.B	november.B
nationwideB
legitimateBlaunch.Bjanuary.Biue,Binteret.Binnovation.BinfluxBin.BichuanBharkBgatewayBgamblingBforum,B	featuringBeveningB	eptember.BengineBengageBeleBdiplayBconcern.BcompetitiveBclearlyBcharityB
bureaucratBbitcoin:BbackingBatomicBapiB
whitepaperBuualBtorBthomaBrollBreturnedBrequirement.B	remainingBraceBpureB
proceedingBplayedBpeerBout,Bonline.Bnft,BmayorBlaunch,BlagardeBjournalBittingBinformation.BimpoibleBidechainBgeorgeBfrozenBfireBentryBecurity.B	ecuritie,BeaeBdownwardBdotcomB
detinationBdekBdealerBdate,BcuriouBcrypto-economyBcrowdfundingBcountyBcopeBcoolBclimateBcientitBbroadBbottomBblock,BbitflyerB	benchmarkBbaiBay:B	accuationB	york-baedByangBwriteBwork,BwildBwhiltBwealthyBvinnikBverifyBverifiedB
valuation.B	uperviionBup,BtreamingBtranfer.BterrorimBroundup:Brik,Brie.B	revealed.Breult,B	reported,B	repondingBparliamentaryBother,Boftware.BmyteryBmuk,BmexicoBleakedB	kong-baedB
inflation.Bin-depthBimprovedBhouingB	generallyBfahionB
exchangingBebangBdefinedBconultationBcapitalization,BcBbillion-dollarBauditBapril.Bannouncement.BairdropBadoptingBadminitratorBaction,BachievedB5,000B201chaB2011B(eh/)ByellenBwright,BwirexBvulnerabilityBvladimirBvcBvaultBubBtoringBtimB
threatenedBtemmedBreumeBreerve,B	real-nameB	quarterlyBpyramidBpullB
predictingBpower.Bpolicy,Bpoint.B
peudonymouBoperationalBmyteriouBmarathonBmanner.B	kazakhtanBiotaBignificantly.Bhow,Bhere.BforenicBflaghipB	facebook,BexpeniveBellerBdrawBdramaticBdefaultBdamageBcryptographyBconiderablyB
conecutiveB	commentedB
china-baedBaxieBaumeBattackedBamidtBall.BaertBaemblyB	adoption,B31B2008B(ir)B(ec).B$15Byork,Byet,BwrappedBwikileakBundergroundBuiteBuddenBubmitBtrendingBtrackerBtoleBtock-to-flowBtextBtetedBruhBrobutBremovingBrelationBquetion.Bpurchae,BprovenBproperB	promotionBpoible.BplungedB	performedB	pecializeBpecialitBpaper,B	pandemic.B
open-ourceBolveBnewdekBmandateBlaundering,BjailBilamicBidBholding,Bhigh-profileBfoundation,BforecatBfeltB
excluivelyBenhanceBendedBelementBdonateBdigital,B
democraticB	defendantBcrime.B	convertedBchnorrBcap,B	avalancheBarrayBamendedBalexBairportBafetyB(ipo)BzhaoBwireBwedenBwaterBviralBunfortunately,BtwiceBtrutleBtitleBtellingBtellarBtartup,Breport:BrecoverBpool.Bpoint-of-aleBorganization.Bnext-generationBmeaningBmaximumBmainlyBlit.BliceningBilverBhitory,BhitoricBhappyBfailureBfacedBexpoedB	expectingB	detailed:Bcrii,B	convincedB
containingB
competitorBcoinbae.Bcode.BchargingBblockedB
argentina,BaimingBafrica.B	affectingBaddree.BadamB(otc)B(etf).B(cbn)BwomanBtweet,Bthere.Bterm.Bripple,Bpropoal,B	procedureBproce,BprepaidBpool,BpoB
permiionleBperiod,BpatrickBowningBource.BoddBoccurBnemBnation,B
millennialBmattBlightlyBlibertyBkey.BkevinBjihanBjayBinvetigatorB	internet,Binfratructure.B
indicatingBhoweverBhittingBgrowth.BgrandBgeminiBfuelBfoxBformerlyB
federationBexceptBevereB
etablihingBeh/BearlietBdraperB
developer.B
demontrateBcryptophereBcroB
conequenceB	commiion.BcoinmarketcapBchatB	certainlyBbountyBanthonyBagB(po)B(imf)B(fa)B(dnm)B$20,000BxBwappedBviitBue.BtvlBtrend,B	temporaryB	techniqueBtate-backedBrun.B
repreentedB
quadrigacxB
phenomenonBoutlineBoutlet,BounceBmillionaireB
mercantileBmember.BmalayiaBlookedB
kyrocketedBite.Binvetigation.BinpiredBincBimilarlyBgermany,BfxBfunBfueledBforbeBfirt,BfinedBfiat.BexceedBever.B	embracingBedwardBdraftingBdown,BdojB
cryptopunkBcoverageBcontributionBconitBcompriedBbarrierBay,BamlB	alvadoranB201cnoB201cforB00e1B00a0iBwilliamBwihBwhomBweakBvietnamBunleBtranactBtetnetBtetimonyBtemporarilyBtealBrubleB
reputationB
republicanBreleae,BrapperBplentyBpathB
partneringBource,BopeneaBon,Bofficer,Boffice.Bo.BnacentBmillerBmeengerBmatter,Bmark.Bkong,BjutinBjune.B
incredibleBigningBhyperinflationBhaltBglanodeBgenlerBfundedBfounder,BformalBflawBfeedBextraditionBenjoyB	donation.Bdevice,BdatabaeBctoB
compatibleBcolombiaBcode,BclimbBclarityBcircuitBbroaderBbeliefBaylorBarea,BadopterBaddreingB65B400B2,000B$250ByahooBwonderBwitchBwingBunivereBunauthorizedB	tranmiionBtool,Btether,Brule,BriotBright.BquantitativeB	principleB	principalBpleadedBpeggedBpacexB	organizerBolderBnon-cutodialBnightBneitherBnearingBnayibBminiter,B	memberhipBmarch.BleagueBjoneBinnovation,BimB	illegallyBignalingBhuobi,B	hort-termB	hitoricalBhandlingBhack,BformatBformallyBfitB
explained,Bether.Bend,B
encryptionBdiputeBdanBdah,BcramerB
controlledBconflictBcomicBcombineBclarifieBcity.BchildrenBboatBbefore.BaumedBandboxBaia,BaiB
accelerateB3dB1.5B00a0haB&pBupiciouBupbitB	trengthenBtrend.Btoo.Bthat.BterminalBrichardB	reboundedBrealizedBrange.BpulledBprohibitBprior.BpremierBpointingBplutokenBpandaBover.BoutputBoracleBolarBname.BmixedBmainlandBllcBlit,BlearnedBlawuit,BjumpingBhodlBhireBha-256Bforward.Bexperience.BexcitingBexceBenforceBeneBeemedBdownturnBdimiedB	detailed.BdeployBdeliveryB	cro-chainBcriticimBcrazyB
conficatedBcheme,BcharlieBchange,BcelebratingBbrokenBboomBbirthBbelieverBauthoredBarguedB	adjutmentB
activitie,Bact.Bact,B75B201cwhatB00a0inB$100,000Bzone.Bwrote.BwinningBurviveBupply.BupgradedBtrengthBtrategieB
tate-ownedBreturn.BremoteBrelativeB	recoveredBratioBqualityBpropoal.Bproof-of-takeBpromotedBprintBpolicy.BphraeBpartner.BoutcomeBordinaryBophiticatedB	objectiveBnew.bitcoin.com.BnanoBmorning.BmeetupBlunoBlow.BjanetBinfinityBgreatetBglobe.B	globally.B	fundraierBfellowBfeature,B
executive,Bell,B	conultingBcontructionBcontributorB	containedB
confidenceB
communitieBcommonlyB	collectorB
collateralBclueBcloureBbtgBboxB	bitflyer,BbehaviorB	baketballBarticle,BamauryB37BvotingBviewedBveruBupport,Btweeted:BtrikeBtokenizationBterm,BtephenBtart-upBtandard.Breleae.BrejectBrehabilitationBrecognizingBracingB
protectingB	protectedBpreenceBpokenBplace,BpirateBoffhoreBoccaionB	norwegianBnormalBmutualBmereBmempoolBmaxBmannerBltdB	location.Blife.BjointlyBite,B	internet.B	improvingBhuttertock,BhiringBharderB
guggenheimB	gibraltarBghotBfillBextentB	examiningBeven-dayBegmentBedgeBdriverB	dogecoin,BcurbBcriptBcomptrollerBcointextBchritineBcheaperB	chainlinkBcarbonBcapabilitieBblameBbitmain,BbenBauthoritie.Baugut.B
artificialB
apparentlyB
announced.Banalyt,BaementB	activity,B201cnewB2015.B20,000B1tB(tvl)Bweekend.BwarrantBvoiceBurpaingBupgrade.BunveilBufferBue,BtrulyBtalentBtaleBtackB
retirementBrealizeBq3BpropectuBpreferB	practicalBpot.Bpixabay,BpickedBpetro.BpanteraBoriginBoptimimBongB	obtainingBnumber.Bname,BnaBmeaure,Bmanufacturer,BlegalizationB
large-caleBjonaldB
indictmentBhigh-rankingBhideBheadquarterBgoogle,B
framework.BflightBexplain,Bexpert,BericBequalBdominateBdiveB	decribed.Bcredit:BcouldntBcore,BcongreionalB
compromiedBclimbingBboBblamedBbitpay,Bbitcoin-relatedBbinance.Bbai.BalaryBaccueB201cvirtualB2010.B(rbi),B(ltc),B$8B$10kB$1.2Bwitzerland,Bweb.B	virtuallyBvideo,BvergeBunuualBuch,BrollingBrig.B	repectiveB	renewableBreaon,Bread.cahBpreliminaryBpoiedBplit.BpermiionBpercent,BpenionBparliament.BparallelBorder,B
obligationB	mileadingBmeeting,BmatchB	mandatoryBmBlow,Blicene.BkepticalBiliconBhowcaeBhighlightingB	governingBgeneral,BficalB	february.BfamilieBfacilitatingBexteniveBexoduBeo,B
efficiencyBdollar-peggedBdicloureBdevBdetail,BdegreeB	december.BcryptographicBcrutinyB
correctionB
contentiouBcontantBclarifyB
chritopherBbrieflyBblockchain.comBbitgoBbancoBantonopouloBanction.BaltBaddre,BaccumulatedBacceingB201cmoreB00a0toB$600B$25B
zimbabweanBwouldntBweiB
univeritieBunit,BunconfirmedB	ubtantialBtrongetB
tranparentBtoughBthing,BterahahB	tatement.BtaiwanBrie,BrevolutBregardleBrefuedB	recently.Bq2Bprogram,BpricedBponoredBpinBphotoB	outliningBorderingBoffice,Boctober.BoberverBnyeBnotionB	narrativeBmirrorBmethod.BmemoBlandcapeBkybridgeBintegratingBinformBhurtBhuntBhim.Bhere,Bglobal,BfreezeBextraBengineeringBeffect.BdogBdicoveryBdependBdemocratBdanielBdaily.Bcot.BconolidatedB
conitentlyBcollaborateBclearingBcitizen.B
charitableB	changpengBcentBbypaBbtccBbornBbittamp,BbelongBban.BbabyB	autralia,Baturday.BapparentBamendBaloneB
allocationB
aggregatorBadminitrativeB24-hour.B22.B00a0thiB(fca)B(et)B$40B	yeterday,BwiderBwhitleblowerBupbit,B	unlicenedBtrangeBtranfer,BthreholdBthreatenB	themelve.BtapBtacticBrivalBreward.BrepectBreducedBrecord-breakingBreceionBquietlyBquarter,BprobeBprintedBpotlightBpecificationBpat.BpackBothebyBoppoedBomniBoffer.Bnigeria,B	necearilyBmore,Bmeaure.Bltd.,Bllc,Blife,BitalyBinteneBindictedBhonetBhomelandBhippingBharplyBhahrate.Bhacker.Bhack.BguetBgottenBgavinBfatetBfaB	exchangedBexceededBempireBell-offBeizeBearchingBdramaBdifficultieBdelayBdektopBdefendBdaoBdallaBcycleBcounterBcorrelationBcorporation,BcontextBconitentB	congremanBconglomerateB	concludedB	commiion,BcombinationB
co-foundedBchritmaBcenorhip-reitantBbrazil,BbraveBbloggingBbitwieBartit,BarkBapplieBandreeenBanalyzeB	afternoonB	accountedBabroadB3,000B201coneB2009,B15,000B12.B10.B00a0cryptocurrencyBwrote,Bword,BwindowBwilonBwhateverBwar,BwageBvotedB
valuation,Bupgrade,BultimateBtudieBtorontoBtelBronBreviewedB	retrictedB	reolutionB
relentlelyBreleaingBreideBrefundB	referringBrealmBreactionBproblem,BpluginBpizzaBpioneerBpeculateBpair.Bminute.BmcelroyB	matermindBmachine.BloadedBjonathanBjereyB	ituation.B
initially,BindoneiaBincorporateBhoueholdBharpBhand.B	greenidgeB	graduallyBgametopBflagBfeelingBexemptBexBeth.BecuringBearnedBdumaB	dominatedBdebate,BcuttingBcurbingBcrypto-baedBcrowdaleBcounelBcontrol.B	contantlyB	conortiumB	compliantBcompareBcla.BchritieBchallengingB	cambridgeBbolivarBautomaticallyB	attributeBarguablyB
applicableBanweredBanalyzedBadviedBadmitB
accreditedB36B30.B2.B11.B00a0onB
00a0chineeB	wedneday:Bvolatility.Bvolatility,Bventure.ButxoButilizedBuploadBupetBunregulatedB
univerity,BunemploymentBu.k.,BtycoonBtudioBroadmapBright,BrichetBrefueB
real-worldBranomB
preventingBpotponedBpopBpoedBparticular,Bparliament,B	pandemic,BovereeBoutheatBnoneBno.BnewcomerBmove.BmitakeBmilitaryBmerelyBlong-awaitedBlogoBlibraryB	launched.BiveBinitiative.BinitiateBhipBhallBfrance,Bforum.BexpreBexpectation.BexactBeuro,Belf-regulatoryBdubaiBdrawnBdimiBdicoverBdeclinedBdangerB	convictedBconvenienceBconolidatingBconference,BciteBcharge.B	capitalitBboard.Bbillion)Baway.BarmtrongB72B49B201cveryB201callB00a0japaneeB(ico),B(ecb)ByourelfByork.BworeBwendyB	violatingBviabtcBviableBvalidB	utainableBurvey,BurpaBunlikelyBtock.Btela,Btatitic,BrolledBrevereBretoreBreducingB	recommendBrBquantityB
prohibitedB	plummetedBphihingBpepeBpavelBoutperformedBoperaBohioBmueumBmapBmacroBlumpBlive.BjoephBitelf.Binteret,BinflowBindividual,BindeedBhaltedBgunB	guaranteeBghanaB	fyookballBfriendlyBforcingB	finalizedBfield.BfarmingBfargoBexpeneB	executionBerrorBerikBentrepreneur,BeligibleB	election.Bec,B	dogecoin.Bcreator,BconventionalB	confidentB	concluionBcollectible.BcollaboratingBcoingeekBcleanB
cla-actionBcellB	celebrityBcap.BbolteredBbnbB	blackrockBbitcoin.orgBbetter.Bay.BawardedB
authority,BattendedBatm,BapproachingB	approacheBaideBagency.BafelyB55B34B201courB201catB2010,B(pow)ByoutuberBxiBwriterB	withdrawnBwelledBweb,BwalmartBvpnBvia,BubequentBtraditionallyBtractionBthen.BrupeeBregainedBregainB	recordingBrecommendedBread.BraidBpurpoe,Bpublic,BplungeB	perceivedBpayoutBpaxfulBontarioB
oil-backedBociety.BnovelBnicolBnflB	movement.Bmove,BmixBmicrobtBmeauredBmeariBmaterialB	lucrativeB	leaderhipBlaunderBlatedBkraken,BknewBkickedBjuridiction.BjeffreyBjackpotBinteretingly,BinteractiveBinformation,BidentificationBheetBgroBgridB	governor,Bgood,BfoundingBfiling,BfemaleBfacilitatedBexplain:BeurBetpBeoulBenormouBenglihBengagingBemployedBeditorBecapeBebaBderivedB
decriptionB	decliningBdeciion,BdadBconiderationBconference.BcompenationB
commentaryBcollegeB
collectiveBcollection.BchooingBcapableBbubble.Bboard,B
bittorrentBbitpay.B	belongingBbanking,B	autralia.B
attractiveBattitudeBatm.BangelBallegingB5.B33B2fB	2018blackB00e9chetB00a0thatB00a0ofB00a0ecB	00a0chinaB(theB(nye:B(cbdc).B$12BwoodBwiftBuperB	ufficientB	trillion.B	trillion,BtoutedB
tournamentBtenionBtender.BtelecomBteaBtating:BtackleB
repeatedlyBrenownedBreacheBrampingBraidedB	property.BpromoterBprion.B	preferredBpolledB	outbreak.BoutageBoccerBnumber,Bnow-defunctB	million).BmetropolitanBmedianBlondon,B
liquidatorBlieBlideBlater.BjimmyBirihBinterviewedBinitedBindividual.Bin,BimportBimplementation.BilbertBhotel,BhidingBheartBheadingBfuelingBformingBforgetBforce.BfixB	favorableBexponentialBenforcedBecond.Bec.Bdouble-digitB	director,BdetainedBdecentralized,BdebutBdeal,B	cryptoaetB	contituteB	conpiracyBconolidationBcolumnitBclaytonBcarcityBcalvinBcalifornia,B	calculateBbookingBbond,BbloomBbitfuryBbitcointalkBbitcoin.com,Bbefore,BbarryBbandBbaketBback.BarrangementBambitiouBaltcoin,BakonBach,B600B44B4,000B28.B2022.B201chowB2012.B19.B1,500B00a0ruiaB(doge)BwindBwagerBvanBurpriedBurfacedBummaryBuggetionBubject.BubcriberBtueday:BtroubledBtricterB	treatmentBtreatBtranformBtormBthird-largetB	tate-iuedBround,BreumedB	returningBrenewedBreident.BpriorityBpoundBponorBpolicie.Bpain,B	oppoitionBopcodeBonline,BnotifiedBnot.Bneed.B	luminarieBlockBlernerB	legalizedBkong.BinnerBincluionBimpreiveBieoBhowever.Bholder.Bholder,BhavenBhalving,BguyBgoal.B	globally,BgangBfriday:Bforward,Bfiat,B	exitence.BeuropolBetcBeducateBduneBdrawingBdemontrationBdefunctB
definitelyB	decribed:Bcryptocurrency-baedBcrucialBcreator.BcounterpartyB	copyrightBconfuionBcomplainingBcnbc,BcertificateBbuBbitcoin.com.BberkhireBbeatBback,BappropriateBapplyingB
aociation,BalphabayBaeBadvierBadaB800B3.B
2018cryptoB120B12.5B
00a0cryptoB(rbi).B	yeterday.B	wikipediaBwantingB	vietnameeB
venezuela.B	undermineBukraine,Budt,BuddenlyBubredditBtweetingBtumbledBtrieBtoppBthing.BtaxableBtartup.BtBreturn,BrelyBrecognitionB	recipientB	qualifiedBpuzzleBpre-aleB	practice.Bpower,BpolihBplethoraB	phyicallyB	perpetualBperpective,BpenaltieB
pecializedBpanicBpairingBpage.BoroB	operator,BolidxBnuclearBnoopBnaphotBmulti-currencyBmonday:BmixerBmaybeBman,BmalayianBmaintainingB
maintainedBlot,BloadBliting.B
litigationB	literallyBliftedBkellyBirelandBipBinvoiceBinventedBintentBintanceB	ingapore,Bindex,Bincome,Bidea.BiceBheet.BhbcBhare,BhardhipBhanghaiB
guaranteedBgrowth,B
generatingBfoundryBfairlyBexplain.Berie,BenuringBdrop,B
dominatingB
diverifiedBdenmarkBdemand,Bdebate.BdbBdanceBcultureBcreekBcoupledBcontent.B
confirmingBcompiledB
compellingBclaifiedBcirculation.BcarryB	campaign,BbzxBbulkBbuilt-inBbrief.BbayBban,B	approval.BandreenBamongtBalone.BabueB6.25B54B201cyouB2008,B20.B(kyc)B(et).B$50kB$2.2Byou.ByntheticBxrp.B	wonderingBwithdrawal.BviewerB	veriblockB	uzbekitanB	undicloedBud)BtypicalBtudyingBtripBtrengtheningBtreet,BtrainingBtool.Bto.B	thereforeBthere,BtheorieBtaiwaneeBreward,Breview.B	requiringB	repoitoryBreplacedBreembleB
reearchingB	quetionedB	purportedBpropoingB	promptingBprofit,BpolygonBpolandBplacingB
phenomenalBpermitBpaxoBpaintingBoweB	operator.Boon,BomewhatBobviouBnode,Bmorning,Bmonero,BmithBmitBmadBm.Blp-baedBlongetB	litecoin.BlendB	legilatorB	launched,BjourneyBixthBinvitedB
intrument.B	integrityBintantlyB
intallmentBinitiative,Bincreae.Bincome.Bince.Bhim,BhenzhenBhairBhadowBgreatlyBgBfantayB
facinatingBevadeBeuro.BetoniaBenjoyedBenergy.B	election,BeentialB	economic,BdividedB	directionBdeputieBdeign,BdatedBdB
cybercrimeBcupBcrowdBcorp.BconvinceB
contractorB
concerned,BcompoundBcollapedB
circumventBcircle,BchoeBchileBcharacteriticBcenarioBcareerBbridgewaterB	borrowingBborrowB	bolteringBblockingBbitfarmBbchdB	baically,Bauthoritie,Baug.B	approvingB
appreciateBamdBaltcoin.Baia.BabraB53B25.B24.B201cthatB200,000B00e9becB(egwit)B(aml)B(B$60B$150BzaifByearnBxapoBweightBwatchedB
wahington,B
vulnerableB	voluntaryBviitedBvegaBurpriingBupperB	uncertainBubpoenaBu-baedBtwelveBtrollB	triggeredB
tranactingBtotal,BtickBthurday:BthomonB
territory.B	telegram,BtandingB
revelationBretriction.BremovalBreidingBreality.BrangingBrageBproprietaryBpeoBpcBpairedBowner.BolidB	official.B	official,Boff,BnarcoticBmoonBmoment.BmnuchinBmergerBmedicalBmarhalBmakerdaoBmailingBluckyB
liquidity.B
liquidity,Blimited,BlegendBkiokBkenyanBjuly.BiraBimpoingBidealBhuffleBhop,BhiftingBhavedBgoldenBgain,B
frequentlyBfoundation.B	follower.BfloodBfinneyBfetivalBfateBexternalBexecutedBexecuteBexaminedB	evaluatedBevaionBeu,Berc-20BequitieBenforcement.BenactedB
employmentBellipticBeliteBeffort.BecrowBdocumentationBdocumentaryB	document.BdiplayedBdetail:Bdepoit,B	defraudedBddoBdcBdankeBdangerouBcorrepondingBcopyB
convertingBcontributedB	continue,BcontingencyBcontactBconcludeBcoloradoBcoinhareBcoinexBclaeBcatchBcanada.Bcalifornia-baedBcabinetBburgerBbroker,BbradBbodieBbe.BbarclayBbadgerBawfullyBavivBautrianBatohi.BarrivedBarmyBanction,Balvador,Balternative.B
algorithm.BalarieBagendaB	aftermathB	activatedBaaronBaangeB8.B30-dayB21.B
201cpeopleB2018theB2011.B14.B1.1B(ln)B(aBzhuoerByou,BwifeBwarmBwannaBviruBupideBupendingBubject,B
ubcriptionB
therefore,BteppingBtender,BtearBtationBtandard,BrujaBroyalBrideBremarkedB	reductionBreality,BquitB	province,BprofitabilityBpoliticBpleaeBparty.Bopportunitie.B
onboardingBolelyBnorwayBnode.BnobuakiBnobodyBniceBneoB	movement,B	meme-baedBmegaBmaterBmaniaBmachine,BloveniaBlotteryBliuB	lithuaniaBliquidationBlifetimeBlicene,BkakaoBjaonB	inventionBintercontinentalBingapore-baedBindependentlyBindeed,B	incumbentBin-gameBhutdownBhottetBhiredB
guideline.BgradeBgarzikBgarneredBgamerBforkingBfinanciallyBfiat-to-cryptoB
excitementBexaminationBevolvingB	everyone.B	enthuiat,B
engagementB	elizabethB	elewhere.BdypBdynamicBdo.B
different.BderiveBderivative,Bdepartment,BdenieB	demandingBdemandedBdelayedB	deignatedBdai,BcreenBcraftedBcourt,Bcot,BcornerBcooperativeBconB	computer,B
commented:Bclaim,BchritianBcheckingBchattedBcharterBbittrex,BbarelyB
auctioningBarBamterdamBag,B
activationBaccuingBabc,B50,000B2xB201careB	2018atohiB130B10thB	00a0indiaB(xrp)B(orB(eth).B(cbdc),B(aic)B$100kBztorcBzone,Bzcah,ByouthByellowBwitzerland.BwitchingBvoorheeBvirginBvectorBurelyB	uncoveredBuk-baedBtwo-dayB	trendlineBtrainBtraightBtorage,Btokyo,BthrivingBtexa,Btaxe,Btatitic.BrouteBreponibilityBrememberBreformB	queenlandBpreentationB	practice,BpopulouBpocketBplaguedBpecializingBpaypal,BparadigmBpakitanBpage,BovertockB
outtandingBoutlet.BnewetBnew.bitcoin.com,BmeteoricBmarBmanipulatedBmaeBlawyer,Blawuit.BlarryBknow,BjudgmentBiran,BinteractBinputB
inevitableB	imilarly,BimagineB	ilvergateBignoreBhotpotBhot,BhopitalB	hollywoodBhiveBheatedBhaltingBhahingB	hackathonBgeorgiaBfreedom.BexpoeBexplodedBexitedBetate,BetabliheB
endorementBeaBeBdramaticallyBdfe7BdeterminingBdelhiBcreenhotBcorp.,Bconumer.BconumeBcontrollingB	contenderB
connectingB
confirmed.B	computer.B	comparingB
committee,Bcodebae.BciphertraceBchildB	chairman,Bcenter.B
capitalizeB
calabilityBbuinemanBbritainBboundBbook.BbitpointBbenefit.Bbch-baedBawardB
attention.B	although,BaitanceBadvertiementBaddree,Badded,B66B6.B18.B00a0announcedB0B(et),BxbtBwyomingBworryBwarning,BwahBvoicedBviolatedBview,BvictoryB	upporter,BupholdBunocoinB
unexpectedBunclearBummer.BuieB	ubidiary,Btweeted.Btrategy.BtranformationBtotal.Btory.Btorage.BtopicalB
tokyo-baedBteppedBtendB
takeholderBtaBroad,BriaBrevenue.Breplied:BreplayBreolveBreearch.Brecord.BralliedB	r/bitcoinBprovingB
proponent,Bpolicie,BpokerB
peronalityBpeirceBpaionateBpacificBoutrightBnoted.BnikkeiBnicheBngB	multitudeBmulti-ignatureBmodeB	migratingBmethod,BmetavereBmarvelBmarkedBlummiBloomingBliableBlevel,Blegilation.Blegilation,Bledger,BleakBleader,Bize.BinviteBinvetigativeB
interview.BinrB
inflation,B	indutrie.BindiceBincorporatedBiland,Bhoue,BhillBheterBheatingBhateBhalving.BhaanBgrow,BgirlBgenler,BgdaxBfundtratBfunding.Bfrom.BfraudterB
forfeitureBervicingBenhancedBemphaizeB	eliminateBelf-regulationBeion,Becure,Beaon.BdonatingBdip.B	diligenceBdepoit.BczechBcreated.BcrackedB	converionB
convenientB
contributeBcontetB	congetionB
concerned.BcommitBcommenceBcome.B
collectingBcoatB
challengedBcenturyBceaedBcategoryBcalled,BburdenBbrief,BboomingBblowBbitgrailB	bitfinex.B	bipartianBbailBayreBavenueB	auctionedBat&tBapplication-pecificBantiviruBanother.B
anonymity.BairlineBaintB
agreement.BacknowledgeB7.B7,000B47B42B4.B2018blockchainB1:1B00a0coinbaeB(ltc)BzebpayByuan,ByoungerBxmrBwithdrawal,Bwith.BwalkBviualButcBummer,BubmiionBturmoilBtuckBtruthBtrongerBtriggerBtreatedBtravelerB	trategie.BtraffickingBthrilledB
tep-by-tepBtefanBteamedBtaxedBtage,BrevolutionaryBrepreentationBrepoBreligiouB
reiteratedB
reearcher,B	rebrandedBrayBquetion,BquetBpuruingBpurportedlyBpumpedBproudBproneBprize.BpreerveBpre,BpraiedBplayboyBphereBpeudonymBpayrollBparty,B	overnightB	overeeingBoteniblyBoleBold.Bociety,BobligedBobcureBnote,BnicholaBmulti-levelBmuicianBmexico,BmexicanB	merchant.B	merchant,BmercadoBmegawattBmarhallBmalayia,BmaivelyBlocalbitcoin,BliraB
legalizingBlee,BkidBkewBkeierBjeremyBinvetigation,BinteroperabilityBinquirieBincomingBinc,BiiBidewayBidentifyingBhodlerBhip-hopBhintedBharhBhandedBgood.BgocryptoBgiant,Bgdax,Bform.BflockingBfeedbackBexpoBexplanationBeventhBevaluateB
ettlement.Bet.B
end-to-endBelectricity.B	economit,BecondaryBearthBeagalBdividendBdippingBdiamondBdepoitedB	dependentBdenyBdelitBdeeperBdanihBcynthiaBcreditedBcorpBcoordinatedBcontributingBconervativeB
completingB	coinquareBcoinjoinBchip.BchileanBcenter,BcandleBcameraBcahaaBburnBbtc.topBbreakoutBbounceBbitwageBbegan.B
available.Barticle.B	appearingBapparatuBanweringBantpoolBannuallyBanalyi,BamcBamazingBakaBage,BaertingBacceleratedB700B58B201ctheyB201cbigB
00a0reportB00a0itB	(formerlyB(2f)B$50,000BwelcomedBweightedBwechatBwbtcBwalkthroughBvulnerabilitieBvoucherBvitalBvillageBupplierBunlockBunlawfulBuk,BtudorBtruckBtronglyBtrickBtreeBtreaureBtravalaBtracedBtraceBtoppingBtolen.Bthreat,BtheyllB	thailand,Btether.B
roundtableBrocketBrockBripple.Brig,BrickBrevolution.B	reviewingB	requetingBrepectively.Brelated:B
regiteringBreceion,Branking.BrakedBpychologicalBpurredBproliferationBpro-bitcoinBportal,Bpopularity,BpoeionBplotBplayer,B
perceptionBpencerBpeloiBpaper.BpaiveBown.B
overeigntyBovercomeBon-iteBold,Boftware,B	off-chainBoaringBoarBnext.B
netherlandBmonexBmonero.B	migrationBmeauringBmeage.B	maximalitB
matercard,B
martphone.BmartbchBmarket.bitcoin.comB
maintream.Bliting,Blibra,BleuthBleonB	launderedBkey,BjulianBjonBjohuaBjohBjob,BjiangBit?B	involved.B
interface.B	injectionB
incrediblyBibmB
hydropowerBhome.BhipmentBhellB
heightenedBhealthyBhathawayBhapeBhahrate,BhadyBgrow.BgriffithBgox,BgearingBfreezingBforthB
financing.Bfiling.BfewerB	exploitedBexpireBexceiveBevolvedBenticeBemployerB	employee,BemployB	emergenceBembeddedBeanBeagerBe-nairaBdraftedBdormantBdmitryBdiverifyBdiruptBdecreaeBcutody,Bcuban,Bcrypto-to-cryptoBcrypto-focuedBconequently,B
condition.B	compromieB
complainedBcommunitBcommunicateBcommoditie,B
commentingB
citizenhipBchart,Bchanged.BchamberB
challenge.BcenorBcartelB	campaign.BcahingBburningBbotonBboldB
bloomberg,BbailoutBbaeballB
backgroundBavalonminerBatohi,Bart.BarmaniBarcaB	arbitrageB
aociation.BalgorithmicB
affiliatedBaferBadviingBadminitration,BaccompaniedBabenceBaaveB[B95B57B46B41B201cwithB201cmotB201cmakeB2012B2.5B16.B140B10xB$1.7BzugBynonymouByearlyBwright.BwithdrawingB	whatminerBweek:BwalltreetbetBviitingBview.BvaccineB	upporter.Bupply,Bupdate.Bunocoin,Buniwap,Bucce.BuarezBtyleBtwitchBtruggledBtron:B	treamlineBtreamB	tranlatedBtranferringBtonyBthreateningBthi.B	taxation.Btax-freeBtatu.Btate-runB	tability.Brun,BrootBriverBridingBrepone,Breource.BrentBremindBregardedBrecord,BreboundBread,Branking,BrampBpurueBpullingB	prototypeBprotection,BproperlyBpromotionalBproceor,B
preventionBpreingBprBpornhubBpolyBpolitician,Bpolice.Bpoition,Bplayer.BpierceBpickingBphone,Bpeculation.Bpat,BparkingBpaportBpain.BpaeBpaciaBovrBoffeniveBobervedB
northboundBnobelBmulimBmoralBminute,BmileBmiBmeritBmeage,BmaverickBmatthewBmathematicalBmanipulatingBmandatedB
maintream,BlidingBledgerxBlayBlaidBlaborBlBkobayahiBknowingBkenya,Bjob.BjeeBj.Bitelf,BintallationB	ingapore.BimplifyBiland.BhybridBhtcBhouldntBhoue.BhopedBhelp,BheitBhamphireBgrowing,BgrantingBgainerB	formationBfluctuationBfeverBfederation.BextraordinaryBexperience,B
everywhereB	everybodyBetoroBergioB
equipment.Benvironment,Bentitie.BenthuiamBenough,BemergeBdurovB
downloadedB	diruptionBdiaterBdiagreementBdetroyedBdepartment.BdefiningB	declaringBcrime,BcorruptBcorporation.Bconumption.BcontructBcontrol,BconfidentialB	componentB	compenateB	colombia,B
coincheck,BcobraBco-founder,BclaimantBcla,BchromeBchannel,BchampionB
chainalyi,BcfoB	certifiedBcentralizationB
celebratedBcarceBcale.BcahleBbrexitBbreadwalletBbradyBboredBblindB
bitcoiner,BbinaryB	beautifulBbatchBballBaway,BauthorizationBauthor,Batlanta-baedBarguingBarenaBarabBapple,BantiguaB
analyticalBalone,BalBaitBairedBaggreiveB
aggregatedBaffair,BadminB[and]B900B30,000B27.B26.B21tB	201ctodayB201coverB201clongB2011,B13.B00a0anB/B(udt).B(fc)B(chapterB(bv)B(bu)B$900B$7Bzero.BwozniakBwineBweighBwatheB	watchdog,BviolentBverdictBurbanBunwriterBunion.Buncertainty,Btwo-yearBtweet.Bturn,B	turbulentB	tremendouBtrackedBthriveBthinBthemeBtage.BroutingBrolloutBreult.BretainB	republic,Brep.Breource,BredditorBquare,Bq4BpumpingBpuheB	province.B	projectedBprogre.Bproceed.B	privatelyBpreentlyBpowell,BpornBplay.Bpecifically,BpeacefulB	patientlyBpartie.Bpark,BoverwhelmingBoutflowB	op_returnBnourielBnickBmountingBmoon?BmogoBmitigateBmetamakBmergeBmark,Bmanager.Bmainnet.B	maachuettBlukeBltc.Blot.Blong.Bloan.Blo.BlloydBlive,B
liquidatedBlionBlimit.Bleat.Bleat,BlahingBlahedBkiyoakiBkeynoteBkeeneB	intitute,BintelBingerB
indicationBhydraBhopperBhigher.BhatBharmBhacker,Bfree.BfranciBfortreBfollow.BfolkBflurryB
fliptarterB	firt-everBfinihedB	financierBfinalizeBfcBfarm.B	extendingBexpene.B	exemptionBevolveBerie.BentralB	enthuiat.B	engineer,BempowerBembracedBeh/.BedtB
drivechainBdowBdo,BdivereBditinctB	diruptiveBdigitBdetectB
deploymentBdenialBcurveB	cryptopiaBcrowdedBcorrectBcontemplatingB	conceivedB
commiionedB	colombianB	colleagueBcoatingBcloneB	citigroupB	cientificBcheduleBcertificationB	caribbeanBcandalBcaaciuBbybitBbut,BbrockBbreadBbouncedBboereB
bet-ellingBbangkoBauthenticationBauction,BarizonaB	approach.BappetiteBapeB	apartmentBanimatedBalready,BalibabaBagedBafeguardBadvieBacknowledgingBachievementBacceibilityB9.B85B8,000B74B69B63B56B29.B201cwhenB201cfirtB17.B00b4B0.1B(fa),B$20kB$19,600B$170B$14B$13B$11Bzug,Bword.Bwar:Bwap.BvenmoBvalley,BurpaeBummonB	ulbricht,Budc,BubhahBtreet.Btranparency,B	tranactedBtrailBtoucheBtory,Btopic.BtokenizeBto,BthielBteepBtarted.BtanzaniaB	taggeringBrumoredBringBrewardedBrevolution,B
retrictingB	republic.B	reluctantBrefuingBreferralBrarelyBpubliheB	publicityB
proviionalBprovedB
proportionB
proponent.B	property,BprohibitionB	progreiveBpricingB
portfolio,Bpopular,Bpoible,Bphone.BpeakedBpateBpart,BpalBowner,BoverviewBolvedBoffline.BoffetBoccBobjectBobfucateBnote:BnormallyBnoieBnexoBneutralBneighboringBnamingBnBmogulBmodifiedB	moderatorBmiami,B
memorandumBmeeting.BmbBmanipulation.BmalletBmall,Bltd,BlouiBloerBlinghamBlettingBletter,Bknow-your-cutomerBkillerBjewelryBize,B	inveting.B
intructionBinteractionB	innovatorB
injunctionBinheritanceB	indoneianB	indoneia,BindexeBindex.BindependenceBimminentBimmediately.BightBhurdleBhotileBhopefulBhavocB	hardware.Bhandle,BgreekBgeneral.BgearBgarlinghoueBg7BfrequentBfreelyBforthcomingBform,B	forgottenBforce,BfinlandBfinderBfincenBfigure.BfifteenBfarmerBfarm,BfactoryBevidentBettlingBetonianBenitiveBenforcement,B	employee.Beffect,Bditrict,BdieBdicuion,BdelitingBdecreaedBdecentralizeBdalioB
cryptophylBcrazeBcramer,B
corruptionBcontitutionalB
conolidateBconfuingBconcentrationBcomplicatedB
committee.BcomfortableBcoffeeBcodyB	charteredBcaterBcatchingBcarolinaBcarnageBcagayanBboyBbonuBbitpandaBbitmain.B
beneficialB	believingBbelgianB	belaruianBbanknoteBavoidingB	augmentedBarmchairB
appearanceB	anctionedBamunBamaedB
algorithm,BalexeiBaitantBagencie,B
affordableBadvior,Badd:BactedBacknowledgedB	achievingB[the]B93B83B68B500,000B3.5B25cfB250B23.B2026]B201cwhyB201comeB201cignificantB2009B00a0areB00a0aloB(overB(ifp)B(fatf)B(etp)B(aum)B$35B$1.1ByoudBwrong.Bwholly-ownedBwhollyBvoluntarilyBvocalBvinBverion.BurvivalB
univerity.Bucceful,BtulipBtudy.Btron,Btreaury,BtranmittingBtranformingBtourimBtourBtop.BtinyBthieveBtexa.BteemitBtarkBtappingBtanfordBtak.BtailoredBroughBrizunB
retitutionBreetBredeemedBreddit,B	reaonableBrallyingBr/btcB
provincialB
proecutionBpreent,BpotalB	poloniex,BpolicymakerB	poitivelyBplit,BpledgedBpleaBplaticBphilippine,B	permanentBpay.BpatronBparticipant.BpanBpamBopinion,BopenlyBolBoccaion,BnyB
non-profitBnegotiationBnancyB	municipalB
multi-yearBmnemonicBmlbBminorityBminorBmemoryB
meanderingBmean,BmarcoBmaleBmaker,B	magnitudeB	magitrateBlinkedinBlegalityBlegB	launchpadB	language.Bkleiman,BkitBkilowatt-hourBkievB
journalit,BjackonBitem,Bir,Binfratructure,BinformalBimmenelyB	ignature.BifpBianBhype,BhurtingBhunterBhufflingBhearnBguardBgovernment-ownedBgolixBgo.BglimpeBgdpBftx,B
freelancerBfoterBforklogBforeverBfinding,BfilecoinBexploionBexpertieBexperimentingB
exhibitionBexercieB
executive.Bet,BequippedB
encouragedBemiionBemail,B
education,BecurelyBeatbchBearlier.BdutBduma,BdotBditchBdiappearBdex,BdevaluationBdetroyBdetinedB	detectiveBderibitB	departureBdenyingBdefyBcycle.BcybercriminalBcrecentBcourierBcotlyBcontinuoulyB
continent.BconjunctionBconequence,BcnnBclaificationB
circulatedBcharge,Bcbdc.BcaymanBcarryingBcar,BbouncingBbitboxBbellBbeijing-baedBbcoinBbcBawaitBaverage.B
authority.BarrivalBarcaneB	approved,B	appealingBapartBaignedB
agreement,Bagencie.B
afternoon.B	affiliateBaffair.B	afe-havenBaertionBactivateBacquiition,B@bitcoinB61B6,000B350B2mbB201cjutB201chighB201cfinancialB201ccryptocurrencieB201cB2012,B110B00a0uB(xmr)B(fc),B(dlt)B(cboe)B(ar)B$80B$75B$700B$2.6B$1.3Byuan.B	world-claBwing.BwiheBwhoeverBwhile,BwheneverBweaponBvoyagerBviolenceBvinnyBvetBverion,B	validatorBurlBupertarBuniformBunifiedBunderwayBugandaBucceededBubtituteBtwentyBturnoverBtrue,Btranparency.BtouritBtotalingBtopic,BtimothyBtimelineBterraBtelecommunicationBteachBtdBtayingBruckBrikyBreviionBreveralBreuter.Brequirement,B
regulated.BregitryB	recurringB
rebrandingBrapidly.BrandomBrampantBrameyBquerieBqueezeB	quantitieBpubliherBpublicly-litedBprotet,Bprotection.BpropagationB	prolongedBprohareB	producer,Bprivate.Bprivacy-focuedBpreparationBprecielyBpowBpottedBportal.BponderBpolice,Bplay-to-earnBpivotalBpilipinaB	philoophyB	peterburgB	peronallyBpermanentlyB	partiallyBpair,BoppoiteBoffer,B
novogratz,Bnotice.Bnoted,Bnot,BnordicBninthB
networkingBneed,BmotorBmoney-launderingBmigrateBmiedBmichiganBmcdonaldB	marijuanaBmany,BmanufacturedBmalta,BmailBlimitingBliftB	legendaryBlegal,Bledger.BlebanonBlebaneeBlandmarkBlabeledB	kontantinBkickBkeenBjohnonBituatedBincreae,B
inception.B
inception,BignatovaBidleBhoreBhold,BheelB	hardware,BhanaBhalBhahtagBhacked,BguildBgrowing.BgrinBgovernBgermany.Bgeorgia,BgabrielBfunctionality.BfulfillBfalloutBfall.BfaithBexpreionBexpreingBexplorationBexit.B	excellentBethicBerioulyBequallyBeportB	enforcingBencouragingBelewhereBegyptBeffort,B	economie.B	economie,Becond,Be-moneyB
draticallyBdorianBdicuion.BdiabledBdeterminationB	detailed,BderekB
deliveringBdelitedB
definitiveBdefi,BdeeplyB	decendingBcryptophere.B
crypto.comBcrahedB
crackdown.BcooBcontent,Bcommunication,BcommentatorB	combatingBcoloalBcollectivelyBcoinlabBclarkBchoice.Bcheck,Bchart.BchandraBchampionhipBcenteredBcarefulBcar.Bcapabilitie.Bcalifornia.B
calculatedBbuying,Bbuyer.Bbuterin,B	bulgarianBbreachedBbrand,Bborder.Bbond.BblueB
blacklitedBbittamp.B	behavior.Bbe,BbatBbangkokBaverage,BaugurB
attention,B	argentineBare.BarchiveBanticipationBangeleBandyBame.Bamazon,BaliBaicbootBaertedBadvocacyBadaptBacceleratingBabruptlyB4:B31.B2022,B201cmayB2008.B2,000+B12-monthB00a0haveB00a0byB(th/).B(cme)B(ccid),B(btg)B(bi)B(andB$70B$18Bzero-confirmationByoutube,ByfiByachtBworld-famouBwitzerland-baedBwipedBwholeheartedlyBwhich,BweighedBweb3BwazirxB	volatile,Bvideo.BurpluBuptickBupicionBunpentBunion,BunicornBundergoBunbankedBuit.BubprimeBubcommitteeBtxBtumblingBtuckerBtrictlyBtreaury.BtrangerB
trajectoryBto-date.Bthreat.Bthough.B
territorieBtax.Btax,Btart,BromanianBridgeBriBreplyB	replacingBreortBrentalBreitantBreident,Brange,B	pychologyBpyB
propectiveB	profitingBprivate,B	prioritieB	preentingBponorhipB	poitionedB
poibilitieBpodcat,BplugBplanetB
pioneeringBpiece,Bphae.Bperformance.Bpay,Bpaword,BpawnedBoverall,BoppoeBopecBolympicBofficer.BoffchainBoff.BobtacleBobligation.BnotificationBnotice,BnotablyB
nonethele,BnicehahB	naturallyBnation-tateBnadaq.B	multi-aetBmountainB
motivationBmotherBmoqueBmonopolyBmonetizeBmoieevBmimblewimbleBmetric,Bmeme,BmealBmccalebBmcafee,BmartinBmarcB
manipulateBmagillBmaduro,Bloe,BliteracyBlimit,BliechtenteinBlagarde,BkrugmanBjuryBjpyBitem.Biran.BinvetigatedBintergovernmentalBintegralBintead.B	intallingBinjectedBinfoB	indutrie,B	incident,Billegal.BiconicBhyperbitcoinizationBhuffledBhub.Bhub,BhoverBhineBhibBheftyBhawaiiBhaunBhappen.B	halloweenBgymBgramBgovernance.BgodBgeraldBgenB	function.BfudBfrankBforayBflorida,BflockBflipBfledBfinnihB
financial,Bfigure,B
feaibilityBfat,Bexpert.BexperimentalB	exitence,BexertB	exceptionB	exceedingBexample.B	endowmentB	elaborateBeither.Beay.Beay,BdumpingBdubiouBdraggedBdorey,B	directly.B
diappearedBdevotedBdevconBdemoBdecribe.B
decade-oldBdebunkedBdebatingBdapperB	cutomizedBcryptoconomy.Bcrah.B	covid-19.Bcoure.BcopieBconvertibleBconitingBconenyBconcept.BcomputationalBcompliance.Bcompetition,B
comparableBcommiioner,BcodingBclickBclearedB
clarifyingBcitizen,BcenoredBcelebritie,BcatalytBcantonBcalableBburnedBbullionBbroadcatingBbountieBbootingBbiqBbegan,BbarbudaBbalticBbafinBbae.BbackdropBaving.BaudiBartificiallyBarriveBaround,BarbitrumBaociate,BanytimeB
announced,Banalyi.BamouraiBamirBaltogether,BaliveBahead.BagaBaeingBadditionallyBad,BaccurateB
accordanceBaccommodationBacademyB:B99B8btcB60,000B43B38B25thB201cmanyB2019anB2018bigB180B15thB10-yearB1.2B00a0u..B00a0meetB00a0bitfinexB(udt),B(mti)B(gpu)B(fincen)B(bc):B$9B$69B$24B$16B$1.9Bzebpay,Bwith,Bwhole.B	welcomingBwangBwanBurfaceB	unnecearyBuncoverBuer-friendlyBucceedBuberBtrueudB	travelingB
tranformedBticket.BthrowingBthrowB	thailand.Bteting.Btet.Btech.BtcBtarted,BtariffBtaprootBtakedownBtable.BrouxBroundup,BroubiniBrevenue,BretracementB	retailer,BrelyingBreliedBrelianceB
regulated,Breceive,B
reaffirmedBrating,BradarBquickly.BpunditBprovokedB
propoitionBpring,BpreviewBpremier:B	preidencyBpreeB
pre-conenuBpopular.BpoppingBphonyB	permittedBperitentBpellB
peculatingBpeaceBpatent,B	ownerhip.Boutlook,BottoBop_codeBonboardB	olicitingBolearyBoil,B
nevertheleBnetherland,BneedingBnchain,BmontanaBmodetBminimizeBminerdB	mechanim.B
meaningfulB
mayweatherBmanufacturer.BmagicBmade.BloyaltyBlower.Blow-cotBlong,Blondon.BlogicB	location,BlikedBlevyB
legitimacyB
legilatureB	landcape.BlaerBlabelBkyBjudicialBitaly,Biota,Binvet,B
intrument,BinterventionBintegration,BinherentBingB	inabilityBimultaneoulyBimmigrationBignupBignificantly,BhydroelectricBhortage,Bhort,BhorowitzBhop.Bhome,Bhitbtc,B
high-levelBheraldBhaydenBharvardBharperBharmaBhariahB	happened,BhandledB	hahpower.BhaheBguruBgrabBgovernmentalBgovernment-approvedBgo-toBgithub.B
fyookball,B	freelanceB
framework,BfoundationalB
forecatingB	footprintBflowingBfirmwareBfederation,B	extortionBexponentially.BexpoingBexit,B
exception.Beu-wideB
ettlement,BentrantBenergy,BendoredBendleBemotionBemefieleB	ecalatingBeach.BdxBdryBdrug,BdoggBdivideBdip,BdiegoBdemontratingBdelBdeerveB	declared:Bdecentralization.BdeceaedBdawnBdaBcutody.Bcryptocurrency?Bcryptocurrency-focuedB
crutinizedBcruieBcroreBcriteriaB	criminal.B	creation.Bcrah,Bcounty,BcountleBcotingBcore.B	cooperateBconumer,BcolumbiaBcoloredBcollateralizedB	coincidedBcmcBclarificationBcioBcautiouBcarrieB
capabilityBcanceledBcammedBcall,Bby,B
burgeoningBbulgariaBbubble,BbtmBbtcc,Bbrand.BbootedBbondedBblog,B
blackrock,Bbitcointalk.orgBbicBbeerBbangBbagBaveragedBavailabilityB
attetationBattendBartit.BarthurBargoB
approache,B
anonymoulyBagenda.B	advocatedBadvertieBaccommodateB	accepted.Babha,B8mbB700,000B67B64B260,000B20thB201chaveB201cdoB	201ccouldB2018leepingB10,000+B1,300B	00e9chet,B00a0withB00a0quB	00a0majorB00a0bitcoin.comB00a0autralianB(udtB(fca),B(etf),B(eh/).B(ecb),B(dapp)B(cfd)B$850B$6,000B$3kB$30kB$1,200BzuckBzhouBzero-knowledgeByonhapBwu,Bwood,BwitchedBwearB	wealthietBwealth.BvoterBviewingBvehicle,ButainedBuperiorB	undertoodB
undergoingBun,Buer-activatedB	ubpoenaedB	ubidiarieBuaBtwo.BtupidBtuneB	tumblebitB	tructuredBtranportationBtranlateBtradableBtorrentBtorm.BtollB	together.BtnabcBtierBticket,BthwartBthirtyB	telegram.BtechnicallyB	tech-avvyBtated.B	tability,BrogueB	ridiculouBrevolveBreveredBrevampedB
retaurant,Brequet,BreponibilitieBreopenBremittance.B
remarkableBrelieBrelateB	rejectionB	reitance.BreilientBreignB
recoveringBrapBrakutenBradicalBr.B
queenland,BquBpythonBpure.ioBproxyBprohibitingBproduction.B
procedure.B	placementBpiralBpike,BphilBperpetratedBperonalitieB	perfectlyBpecifiedBpartnerhip.BpartlyBpalmerBoperate.Bof.B
obervationBnight,BnannyBmwBmuhammadB
moratoriumBmoothBmoodBminitry,BminimalBmind,BmicropaymentBmicroBmetal,B	meantime,BmcgloneBmatureBmatchingB
martphone,B	manhattanBlureBluhBloweringBlowedBloan,BlinkingBlineupBline.B
limitationBlending,BleapBland,Bknown,BkiBkew.comBkentuckyBkarpele,Bjuridiction,BjameonBinvokeBinterpretationBinternational,B
interface,BintagramBinquiryBinjectBinitingB	informingB
indirectlyBincentivizeBimpactedB	immutableBignificanceBicedBhype.BhyBhumanityBhintBhimelf.BhillingBhikeB
healthcareBhakenBgraceBgovernance,B	giveaway.B	generatorB	futuriticBfurther.B	full-timeBfrenzyBfrehlyBfloydBfloorBfloodingBfloodedBfinihBfiftyBfact.B
facilitie,B	facebook.BeyeingBeyBextractB	expected.BexcueBexcludeBexchange-litedB	everyone,Bevaion,Beur,Betate.Bequitie,BequihahBep.BenBemptyBembayBeizingBedt.BeconomicallyBebatianBeaon,BdroveB	donation,BdnBdiemBdiedB
dicoveringBderivative.BdemieBdeletedBdeignerB
defamationB	deceptiveBdeath.BdealtBdad,BczBculturalBcryptorubleBcroatianBcriticizingBcrieBcredibilityBcovidBcottenBconveration.Bconumption,BconumedBcontrat,BconflictingBconcern,B
completionB
committingB	commencedBcolorBcollateral.B	coingeckoB	clampdownBchivoBchile,Bcaue.Bcardano,BcaracaBcaling.BcalculationBbrower.BbriefingBbrickBbpBborrowedB
bolivarianBboguBblockchain-relatedBbitparkBbitmex,BbitcoinvBbip70Bbelaru,Bbeijing,BbehemothBbch-centricBbccBbarrelBbakkt,BbahrainBbae,Bb.BaviationBaumptionBauction.BatifyBart,BargentinianB	approval,BappreciationBanwer.BangryBamount.B	american.Baltogether.BalterBalike.BalbumBalarmBaidedB	africryptB	advertiedBadjutBactreBactor,BaccruedBaccompanyingBaccidentallyBacceptance.BabundantBabuBabkhaziaB78B52B4thB400,000B3:B300,000B201ctheeB	201cgreatB	201catohiB2019:B2007B2,500B170B16thB1,600B0430B00a0whyB00a0venezuelanB00a0venezuelaB00a0tradingB00a0blockchainB00a0bitcoinerB0.5B-1B(xrp),B(qe)B(ol)B(nft).B(hib)B(gbtc)B(fbi)B(defi),B(cftc).B(ada)B$7kB$60kB$4,000B$32B$120B#bitcoinBzhao,BzaboBzBxlm,BworthleBworrieBwitneingB	willingneBwholealeBwhereaBweepingBwearingBwatch:Bwap,Bvote.BviolateBviibleBvehicle.BurroundB
urpriinglyBupplieB
unupectingB
unreliableBunicefBundoubtedlyBuncertainty.BufcBud).B	u.k.-baedBtuntBtroyBtreed.Btrategy,BtranactionalBtraitBtraffic.BtracingBtoodBtokyo.BtightB	three-dayB
thoroughlyBtherebyBtheft.B	texa-baedB	terminateBtehranBtealthBteacheBtayedBtay.BtaxingB	taxation,BtatueB	tatiticalBtakeoverBtabBrun-upBruellB	revealed,B
retrictiveBrepone.B	reiterateB	reilienceB
reidentialBregBreceiptBreal.Breal,BreadineBrally.B
quetioningBqueenBqeBpurgeBpunitiveBpuertoBprovablyB	propelledBproblem:BprobabilityBpokktBplanet.BpivotBpiteBpiritBphae,BpennyBpeech,B	peculatedBpeachBpaword.BpatreonBpartnerhip,Bover,BoutpaceB	outbreak,BotenibleBorganiationBopponentBone-yearBofacBoceanBnow-deceaedBnovotiBnordeaBnewerBnadaq,Bmulti-igB	motivatedBmoodyBmizuhoBme.BmaximizeBmarkingBmariaBmakBmaintenanceBmade,B	lundebergBlovedBloppBloopholeBlogBllpBlivedBlippedBlimited.B
likelihoodBlifetyleBleerBleague,B	language,Blab.BkyrocketingBkrwBking,BkilledBkenBkeepkeyBjoB	javacriptBjacobBinvitingB	inveting,BinvereBintrinicBintitutional-gradeBinternationallyBinterminiterialBintermediarieBinteriorBintegration.BinflationaryBincurredB	implifiedB
ilverbloodB	illutrateBillegal,Bight.B	identity.B	identitieBidehiftBhutdown.BhootingBhold.BhodlingBhockBhoardingBhigh-rikBhexBheroeBhegemonyBheddingBhawkBharriBhappen,Bhandle.BhalvedBhailedBgregBgreetedBgrapBgo,BgaugeBgarzaB
functionalBframeBforgedBforbidBflourihBfitchBfirt.BfirmlyBfine.BfighterBfear,Bfat.BfamoulyBfactor.B	facility.BfB	extraditeBever,B	evangelitBeurozoneBetonia,Berver,BerntB	entiment,Benough.BenjinB	enhancingBenhancementBene,BenderBenactBelectrumB	eay-to-ueBeagerlyBdumpedBdump,BdrinkBdream,BdreadB	downturn.B	downturn,BdownloadingBdoublingBdoomBdoo,BdonorBdirtyB
direction.BdiableB	detectionBdepreciationB	deployingBdeperateBdenominatedBdelveBdek.B
defraudingBdefraudBdeficitB
dedicatingB	decribed,Bdarknet,BdakotaBdahboardBcut.Bcuba,BcubaBcryptographerB	criticizeBcrippledBcreated,BcovetedBcounterpartB
convictionB	conultantB	continuouB
continent,BcontemporaryBconcept,BcompoedBcommunity-drivenB
commandingBcoming,Bcollectible,BcollaborativeBcointarBcoinone,B	coinecureBcoincideBcoalBco-ownerBclockBcientit,BchurchBchatterB
chainalyi.Bchae,Bceo.Bcentury,BcathieBcarloBcaredBcapeBcandyBcale,BbutedBbug,BbrookBbridgingBbrewingBbrandedB
boton-baedBbonueBbobbyBbloqB	blocktackBblockpreBbithumb.Bbitcoinbch.comB	billion).BbiddingBbeverageBbeginnerBbeepleBbeachBbattlingBbanking.BbalancerBaying.BavalonBautoBautinBauditedBaturday:Bartwork.BanyhedgeB
anticipateBand/orBalanB
airdroppedBage.BaffordBadvertB	admittingBad.Bacquiition.BaccomplihedBabroad,Babout.Ba,B[a]B96B76B62B5nmB32mbB21hareB2022B	201cwhileB201cweveB201coB
201cglobalB201cfromB201cethereumB201cbutB2018virtualB1:B160B12thB11thB11,000B1.9B00f3nB00e1nB	00a0putinB00a0moreB00a0forkB00a0forB00a0bitmainB(occ)B(fa).B(cio)B(approximatelyB(akaB$8kB$6kB$425B$230B$2.5B$2.3B$2.1B$175B$1.4B	zimbabwe,Bzero,BytemicBygnumBx.BwooBweden,Bweb-baedBweatherBwar.Bvietnam,Bviabtc,Bvendor.B
validationB	valentineButility.Burvey.BupervieB
upermarketBuperintendentB
unlimited,BuniteBunfortunateBunealedB
undercoverBummit.BuitedBuicideBuhiwapB	uceptibleBturkey,BtudiedBtrillion-dollarBtrial.B
tranlationBtotallyBtoomeyBtolen,BtoeBtobiaBto-dateBtitanB
tighteningBtigerBthrownB	threhold.BtheoreticalBtheft,Btetnet.BterpinBtenneeeBtedBtech,BteachingBtaughtBtatu,Btated,BtarringBtalebBtactic.B	routinelyBroad.B	rewardingBreveal.Breuter,B	reurgenceBretoredBreorganizationBreiteratingBreignedB
regularly.Bregitration.Bregitration,BredeemBreceion.B
re-enabledBrbcBrandBquetion:B	publihed,BprungB
previouly,BpremiereBprediction.Bpre.BpragueBpoweringB
potential.BportugalBpolygon,B	polychainBpoll,BpolkerBpoliticallyBplay,BpipelineB	pineappleBpiece.Bphilippine.Bphenomenon.Bperpective.Bperon.Bpent.BpectatorBpecBpeak,BpaycheckBpaxful,BpaxBpaul,Bparticular.Bparticipant,BpaengerBpackage.Bp.BovietB
overvaluedBoverturnBoverea.BoutpacedB
organizingBorganizeBopenea,BoonerBomedayBome,BoloBokcoinBocietieBoberveBnowden,BnormBnonfungible.comBneedhamBnatwetBmultiplayerBmultiigBmulti-millionBmulti-billionB	monitoredBmongoliaB	moneygramBmodiBmocow,BmidBmiami.Bmexico.Bmetal.BmetadataB
merchandieB
mentioningBmemo,BmellonB
megauploadBmedium-izedBmedium,Bme,B
luminarie,BloginBline,Blike.BlengthyBlengthBleaderboardBleader.Ble.Blatly,BlamboBkyc,Bkorbit,B	kobayahi,Bkingdom.BjuanBjokeBjerryBjBizeableBizableB	ituation,Bipo,B
intriguingB	intitute.B	inpectingB
ingaporeanBinfuedBinfectedB
inevitablyBincorporatingBincluiveBin-houeBimportedBimpliedB
imperativeB	impendingBimonBimmuneBimmediately,BidexBidea,Bide.BibanBhuobi.BhuntingBhowcaedBhouing,B	horowitz,BholeBhinhanBheet,Bhacked.BgulfBgreeceBgrarootBgodwinB	giancarloBgeneration,BgawBgalleryBfunding,BfrogBfree,Bfound.BformulaBforever.B	forefrontBfoolBfloatingBfinikoBfilerBfifth-largetBfictionBfeetBfaultBfancyBfan.Bfamily.BfalelyB	facinatedB
facilitie.B
extractionB	explorer,Bexperiment.BexpenditureB	expected,B
evaluationB
evaluatingBenjoyingBenigmaBenergy-inteniveBembezzlementB	embattledB	electoralBeighthBdutyBdrop.Bdouble-pendBdoge,B	directoryB	director.BdineyBdimon,BdifferentiateBdifferBdetokenBdepthBdenominationBdenniBdemontratedBdemographicBdek,B
deepdotwebB	decreaingBdecline.BdecidingBdecentralization,Bdealer,BcypruBcryptojackingBcrownBcronjeB	criminal,BcrewBcredBcraftBcouncil.B
correlatedB	correctlyBcoordinationB	conveyed.Bconveration,B
conventionB	contactleB	contactedBconcreteBconcedeBconcealB
complaint.Bcompetitor,Bcompetition.BcompelBcommunication.Bcollection,BcollaboratedBcoinoneBco.,BcoBclub,Bcitie.Bcitie,BcircumtanceBcircle.BciaBciBchunkBcharle,B
challenge,BcentraBcent,BceloBcelebrationB	cautionedB	carefullyB	capacity.B	canonicalBcanaan,Bcammer.Bcah-poweredBbyteBbuzzBbutton.BbrunoBbrowingBbreitmanB	breakdownBboatingBblockadeBbitkanBbitgo,Bbitcoin.org,Bbit,BbiaBbet,BbermudaBberbank,BbenjaminBbelgiumBbeeBbbcBbankruptB	bank-iuedBbackdoorBautriaB	audience.BattributingB	attackingBarrangeBarm,BarchitectureB	arbitraryBapproaching,B	approach,Banother,B	anonymou.BandreBanarchimB	analyzingBampleBamount,Baml/kycBamfBalvador.Bafter,B
afghanitanBafe.BaembledB	advancingBadvancementBadminitration.BadequateBactivit,B	accutomedBacceleratorBabroad.B	abandonedBabB88B86B84B81B59B39B2:B	201cwouldB201crealB201civeB201cillegalB201cdoeB201ccanB201ccahB201cbyB2018cryptocurrencyB2.6B10nmB100xB1.4B1.3B1)B
00a0trezorB00a0popularB00a0exchangeB00a0bchB00a0bankB(tx)B(tnabc)B(pboc),B(nye)B(nba)B(na)B(atm)B$800B$500,000B$33B$2,000B$140B$1.6B
zuckerbergByenomBxinjiangBwonderedBwiquoteBwindow,BwildlyBwild.BwhoveBwhereverBweetBwazirx,Bwarning.Bwa,BvowedBvirgilBviion.Bvietnam.BverticalBverification.B
verifiableBverdeBver,BvaryingBvalkyrieBuual,ButainBurvivedBurvey:B	upgradingBupdatingBunwantedBuntableBunleahedBunfortunatelyBunfairBundertakingBunderpinningB
unanimoulyBukraine.BuitableBuhiBubtantiallyB	ubjectiveBuableBtwo-wayBtuttgartBtumbleBtrump,Btrue.Btreed:BtreaurerB
tranmitterB	traction.Btown.Btoronto-baedBtoring,BthrewBtheory,Bthankfully,B	terminal.BtencentBtecraBteamingBtaxiBtaunchBtarget.Btanley,BruralBrouhaniBroadmap.BrhetoricB	retreatedBrequet.Brepreentative,BrepeatedBrenderBremoved.Bremittance,B	remainderB	rejectingBregretBrefualB	recovery.BreckleBrbzBrationalBrampedBramificationBrailwayBquoineBquebecBpurportBproteterBproof-of-conceptBpromptlyBprolificBprofitable.Bprivacy-enhancingB	prevalentBpremium.B
preferenceB	precedentBpread-bettingBpoterBportingBpopularity.BpoppedBpokemanBplummetBpleadBplant.BpileBpending.BpenaltyBpenBpeer-to-peer,Bpeed.Bpattern.Bparticipate.BpareB	parachainBpanamaBpaed.BpackedBpackage,Bown,B	overight.BoutperformingBoro,BoptedBopioidBolvingBokungBocialitBoccaion.BobvioulyBnydigBnowadayBnext,BneutrinoBnefariouBnature.BnarrowBnahBmountBmortarBmlmBminuBminitrieBmined.Bmid-may,Bmicrotrategy,Bmetric.BmeituBmayerBmartenBmarioBmanualBluredBluno,B	lukahenkoBluckily,BlokBloganBlockingB	lockdown,BlocallyBlocalbitcoin.Blocal.bitcoin.comBlobbyingBllc.Bliquid,BlinuxBlibertarian,B	liabilityB	leverage.Blately.BlamborghiniBkyc/amlBkwhBkitcoBjutice,Bjone,BjohannBizedBirinBirael.Birael,BinviibleB	intructedBinter-miniterialB	inolvencyBinflatedBilver,BillinoiB
ill-gottenB	ignature,B	identicalBicelandBhyperinflation,BhubandBhoutonBhortingBhonorBholyBhockedBhitbtcBhigher,Bhearing.B	happened.BhapingBhandcahBhalveBhalongBhalf.BgtBgrid.BgrayBgrailBgraduateBgbpBgateway,BgateBgargBfunnyB	frutratedBfraudulentlyB	francico,BfrancB	forecatedBfordBfor,BfootBflatBfinding.BfinancedBfinally,BfillingBferguonBfeatBfamily,BfactionB
facinationB
extraditedBexpoure.Bexplanation.B	expenive.BexhibitBexcludedBexacerbatedB
evergrandeBeverelyBevening,Bevaion.Betoro,BetnB
etherdeltaBetc.Betc,B
equipment,BepicBentertainment,B	entencingBenate,B
empoweringB	employingB
emphaized:BemphaiBecalatedBeat.Bearlier,BeamlelyBdyingBdupedBduoBdruckenmillerBdraticBdongB
dominance.Bdnm,BdjBdiviion,BdiverifyingBditrict.BditributingBdifficulty,B
difficult.BdicrepancieBdiconnectedBdiceB	diapproveBdiagreeBdhabiB
devatatingBdetectedBdenmark,BdelegateBdefi.BdefendedBdecentralandBdeath,Bdealer.BdavoBdatingBcyberattackBcut,B
cumulativeBcryptophere,Bcryptocurrencie?Bcrypto-backedB
credentialB	creation,BcreamBcrahingB	covid-19,BcornellBcopB
conultancyB	continue.B	complyingB	complete.B
commodity,Bcoming.Bcoinmarketcap,B	coalitionBcnbc.Bclaic,Bcirculation,BchulmanBchip,BchickenBcheB
change.orgBcenturieBcenoringBcene.BcautionBcargoBcapabilitie,Bcanaan.Bcan.B
calculatorBcafeBcadBcableBbuilderBbug.Bbtc.com,Bbtc.comBbruceBbroker.BbreakthroughBbrdBbrazil.BboriBborder,Bbolivar,B	bloxrouteBblockchain-poweredBblockchain-backedBblockcapBbitoBbitcoin-friendlyBbipBbezoBbet.BberbankBbeleagueredBbeautyBbearingBbch-poweredBbarterBbarredBbankman-friedBbanker,B	bandwagonBbacklogBb,BawfulB	automaticB	authorizeBaumingBatariBarret,BappreciatedBappointmentB	applicantBappeal.B
anonymity,Bannouncement:Banniverary.B	animationBangerBancientBanchoredBanatolyBanarchitBalonzoBalikeBalignedBaggregationBaffordedBafe,BaboluteBa.B9thB98B90-dayB9,000B87B82B77B73B5thB5:B5,000+B40,000B201d)B201ctooB201cprovideB	201cmajorB201cexchangeB201cdecentralizedB201cbuyB2000B2.3B2.2B137B12,000B1,200B00a0publihedB00a0manyB00a0autraliaB(udc),B(nft),B(ma)B(ltc).B(fomc)B(etc)B(ama)B(aboutB$5.6B$5,000B$40kB$12kBzhanByen,Byear-to-date,ByeByahoo!ByBwritBwrapBwordpreBwithdrewBwill,BwidgetBwhimBwhatappBwerentBwateBwaryBwarmingBvirginiaB
violation.Bvinnik,Bvictim.BvibrantBvendingBvariantBvalidateBurgentBuploadedBuperintendencyB	unveilingB
untoppableB	unchangedBummit,BuherBugget.BugandanBucceorBubidyBubidizedBubiBuage.BuafBuabilityB	tumultuouBtudent,Btrx,B
trutworthyB	tructure.BtribeoBtoomimBtoo,B	together,BtiredBtimulateBtickingB	threhold,Bthought.Bthink.Btheory.Bthee,Btezo,Btet,B	terrorim,Btep,BtenthBtart.Btake.Btak,BtaihuttuBrutBruhingBrubinB	round-up:BrobBriing,Breview,B	retweetedBrettigBretiredBretartBrepliedBreplacementBrepeatBrepayBrenderedBremotelyBreminderBremark,Breleaed.BredenominationBrecreationalBreceiverBready,BreadilyBrandomlyBrally,BrajabiBquietBqtumBq&aB	pyongyangBpwcBpurpoe-builtBpublicly-tradedBprofeional.Bprocee.B
procedure,Bprion,B	precedingB	powerhoueBpotterBportnoyBpoingBplittingBplannerBpixelBpikingBpike.Bpicture,BphilipBphere.BpetBperformance,BpennylvaniaBpaypal.BpaveBpartie,BpariBpalmBovrlandBoutpokenBoutlawBoutcome.B
originatedBoppoingBopinion.Bopen.Bop_checkdataigBop-edBonly.BomewhereBomehowBokayBoil-richBoftbankB	occurringBocarB	objectionBobamaBnotaryBnixonBniftyBnichalBnexuB	newletterBnevadaBneeded.Bnature,BnarendraBnamelyBmytery.BmunicipalityBmulti-chainBmultiBmuk.Bmuic,Bmuch.BmooreBmoney?BmomBmoltBmoduleB	million),BmicreeBmeterBmergedB	maverick,BmartialBmar4Bmanipulation,Bmandate.BmahedBmae.BluminoBlowingBlovenianBloomBlonger.BlogiticBliftingBliberalBliBletitiaB	leningradBlehmanBleft,BlazloBlayer:Blayer,BlawrenceBlaureateBlate.BlaptopBlappedBkpmgBkiyoaki,BkittieBkinBjudge,BjoelBjedBjeB	iterationBiterBironBir.Bipo.BintermediaryBintermediarie.BinterimB	intenifieBintelligence,B
influencedB
inflexibleB	infinity,Binevitable.B	increaed,B	imulationBimpreedBimplieB	implicityBimplexBimple,BillBigorBignatovB	idechain,BiaBhugelyBhrinkingBhowardBhorizon,BhookBhomeleBhitoricallyBhippedBhill,BheavyweightBheadwayBhaven,Bhave,BhaterBharryBharmonB	hapehift,Bhalf,B
haktikantaBhabitBgreen,BgovernedBgorenBgone.Bgoal,BgeopoliticalB	geographyBgeekBgamer,BgalacticBfunctionality,Bftx.BfranchieB
fractionalBfounder.B	forwardedB	fortunateBformat.Bfor.BfoottepBfomcB	follower,B	follow-upBfoilB
flourihingBflockedBfleetBflakBfirearmBfile.Bfield,BfiacoBfavoredBfall,BfalconBfairneBfactoBf1Bexponentially,BexploiveBexpediaBexmoBeverything.BeveBeuraianBethicalBerver.Berror.Bequity,BepidemicB	eparatelyBentity,BentailBene.BendoreBencounteredBenate.BemirateBemergentBemail.BemaBelevenB
elaboratedBegyptianBeence,Been.Beed.B
education.B	educatingBecretiveB	economic.BearchedBeamleBdydxBdullBdrainedBdrainBdodgerBdoctorBdltBdkB
ditributorBditribution,BdiperedBdipeneBdipelBdigitalizationBdifficulty.Bdid.BdicloingBdex.BdevereB
detructionBdepreionB	depoitionB
democracy:BdelayingBdelawareBdeign.BdeflationaryBdefiantBdedicateBdecentralized.BdebutedBdebt.Bdebt,B
datacenterBdamage.Bdah.B	cyberpaceB	currency?Bcrypto-marketBcrypticBcroingBcroatiaBcraftingBcpuBcountingB	countdownBcotaBcorrepondentBcorrepondenceBcorp,B
coordinateB
convincingB
contradictBcontitutionB
contituentB
continued:BcontemplateBconficationBconcentratedB
comparion,Bcommon,Bcommoditie.B	commerce,B	combiningB	cointext,B	coinourceB	coingeek,BclothingBclemencyBclear.Bclear,BclaifyBcience,BchartingBchaingB	cenorhip.BcaualB	categorieB	caramucciBcampuBcampBc.Bbuy.Bbureau,Bbukele,Bbtc-baedBbrower,BbrinkBbrandtBboxingBbot,Bbook,Bbolivar.BbobBbnyBbnb,B
bloomberg.B
blockfolioBblockchain.infoBblamingBbitpicoB
bitcoiner.Bbillionaire,B
bifurcatedBbethBberlinBberkeleyBbeggingBbar,BbanqueBbaileyBbai,BaxBawaitingBavvyB
avalanche,BatlantaBath,Barreted.Baround.BarmedBarieBariB
argentina.BarbitrationBarbitarBapphireBappealedBapB	anything.BanderBandbox.BanchorB	analytic,B
ameritradeBame,Bamazon.Balternative,BallocateBalchemyBalabamaBairdrop,Bag.B	affected.Baembly,Badvior.BaccumulationBaccumulatingBacceedBacce.B	abolutelyB[theB89B29-yearB201cwaB201conlyB201cnowB	201chouldB201cdueB
201cdepiteB	201cafterB2018cahB200bB2009.B2.8B1t,B1mbB13,000B1000B00f6reB00a0waB00a0report:B00a0latB	00a0koreaB
00a0formerB00a0fiveB00a0europeanB00a0digitalB(wbtc)B(otcqx:B(ofac)B(mw)B(inB(finma)B(fb)B(f)B(edt)B(df)B(dcg)B(cftc),B$7,000B$500kB$44B$400kB$350B$3.5B$25,000B$23B$21B$2.4B$17Bzug-baedBzelenkyBzclaicBzcah.Byoung,Byield.Byear-over-yearByear-endByaBxmr,BxigmaBx,BwoundBworth.Bwore.Bwore,Bwitne,Bwinter.BwinklevoBwiftlyBwieB	widomtreeBwhy.Bwhitepaper,Bwet,BweptBweirdBwedBwarehoueBvulnerabilitie.BvpBviualizeBvirtueBvipBvictim,B	viabilityBveraBvendor,BvaryBv,Buual.Burprie,Bure,BupwingBuperviorBunretB	unrelatedBunidentifiedBuk.BuaeBtwo,BtroveBtringentBtrezor,BtratiBtoedBtockpileBtobaccoBto-date,BtmcBtirBtimingBtideB
throughputBthree-letterBthiel,B	themelve,BtheaterBthe9B
territory,BternBteemBtaxed.Btart-up,Btanford,Btalk,BtakforceBtable,Brumor,BrtBroutBroll-outBrole,BrocheBripeBrich.BrhymeBrevolutionizeBreviveB	reviewed.Bret.B	reporter,B	repaymentBreorgBrenminbiBremindedB
reignationBregime.Bregard.BrecueBrecruitBrecommendation.BreclaimBrecallBre-openB	rationaleBrating.Brather,BrajyaBrace,BqueryBpvBpuruantB
publicizedBprotetorB	protetingB
propertie,B
promotion.BprofiledBproecuteBproduction,Bproceor.BprideB
prevailingBpreure.B	pretigiouBpremieB	preently,BpredictableB	predeceorB
pre-releaeBpppBpovertyBpopulation.BponoringBpolitic,BpieB
photographBperonnelBperon,B	permiion.B	performerBpending,Bpeculation,BpectrumBpatchB	parameterBpaoloBpantherBpanningBpagniBpacia,BowingB
outperformB	outpacingB	otherwie.BorganizationalBorangeBoptimizationBopportunitie,Boperate,BopaqueBonyBonetBone-timeBoldexBolana,BoccupyBobervingBob1BoakaBnorway,Bnorm.B	newpaper,BneceityBneceary.BncubeBnavigateBnamely,BnaaBmythBmyelfBmybroadbandBmumbaiBmullBmuician,BmoonhotBmonterB	momentum.BmmaBmiueBminiter.BminhBmind.B	miletone.BmigratedBmieBmicrobt,BmetroBmenuBmemorieB	memorableBmemo.cahBmedvedevBmbtcBmarriageB
marketing,BmanionBmaneuverB
malta-baedBmaintreamingBmaintayBmac,BlynBlumpedBluiBluckBloweredB	long-timeBlong-tandingB	lockdown.BliteningBlited.BliquidatingBlilBlike-mindedBlibertyxBlibertieBlennonBleague.Blead.BldBlbcBlatamBlarenBkypeBkrillBkrawizB
knowledge.BknockedBkind,BkillingBkepticimBkbBkanaBjinpingBjeopBjaredBix-digitBiuer,Bitaly.BireBinurerBinuredBintroduced.B
innoiliconBinkB
initiatingB	incident.BincentivizedB	inauguralBinactiveB
inaccurateBin-toreBin-peronBin-appBimpreionBimmonB
imilaritieBileB	ignatova,BidelineB	idechain.BiconBhypedBhungerBhotbedBhot.Bhortly.BhorizonBhootB	homeownerBholiday.BhoardBhkBhimelf,Bhigh-performanceBheziBhenceBhelp.BhedgingBhe,BhboBhardcoreBhard-earnedBhapedBhaipoBguptaBguide,BgroundbreakingBgreyBgreen-lightedBgratefulBgovernment-backedBgoogle.Bgoe,Bglobe,BgeorgianBgate.ioBgarg,BgarbageBfunneledBfungibilityB	function,BfrozeB
friendlierBfray.BfourteenB	formulateBforfeitBfomoBflyBfluctuatingBflorida.BflexibleBflawedBfiredBfingerprintB
financing,B
finalizingBfilipinoBfile,Bfeed,BfbBfargo,Bfactor,B	facility,B
extenivelyBexportB	explorer.B
exploitingBexperimentationBexemplifiedB	executingB	exchange:B	evidencedBevanB	etimatingBetimate.BeruptedBerialBergeyBepteinBenviionBentitie,B	entiment.BenemyBembargoBeluiveBell.Bell-off.BeducatorBecrecyBe.BdutieBdump.BdublinBdreamedBdomain.Bdomain,BditanceB
direction,BdiputedB
diminihingBdilikeBdigital.B
different,Bdicued.B
dicontinueBdiater.BdiarBdeviedB	detractorBdeterminiticB	depoitoryB	democracyBdelay.B	deepeningBdecline,B	decidedlyBdebatedB	deadline.Bdatabae.Bdarknet.Bdai.BdaaBd83cBcurve,BcuratedB	cumberomeBcryptocurrency:Bcryptocurrencie:BcryptocompareBcrypto?Bcrypto-economy.Bcrypto-anarchitB	crutinizeBcritiqueBcreate,B	cramblingBcraheB
crackdown,BcoopBcontrovery.B
contitutedBconoleB	conficateBconenu,BcompromiingB
complexityB	compelledBcomoBcommonwealthBcoinex,BcoineedBcoin)Bco-chiefBclub.Bcloud,BclotheBcloed.B	clamoringBcivicBchipperB	chipmakerBchiff,BchiaBcheck.Bche.comBcharity.BcharacterizedB
character.BchaoBchamathBcentbeeBcene,BceaingBcbB
categorie.BcarrierBcarlBcardonaB
cardholderBcappedBcapitulationBcapianB
candidate,Bcan,Bcammer,Bcall.BcahedBcahbackBcah-centricBbypaingBbunnyB	bundebankBbuenoBbtc1Bbroker-dealerBbrettonBbreach.B
brainchildBbowlBbotwanaBbot.BborrowerBbitwalaBbitcoin-focuedBbitcacheBbitbox,B	biometricB	billboardBbetter,BberwickBbeneathBbelongedBbelieve,Bbehind.BbeamBbattle.B	barcelonaBbar.BbankyBbackupBbackboneBaying,BavidBavedBavatarBauranceBattainBattachedBathleteBathenaBarrowBarena.Bare,B	approved.Bapi,Banyone,Banti-cryptoBantB
announced:BamendingB
all-in-oneBalignB	alejandroBalarmingBakon,Bairline,BagriculturalBagent.Bagent,B
advocatingB	advocate.BadventBadhereB
accumulateB
accountantB
accompliceB[i]B800,000B71B50.B4chanB3xB3rdB3.6B	201cunderB201ctradingB201ctakeB201cregulatoryB201coperationB201cold,B	201cmoneyB201clowB201cinvetorB	201cindiaB201cfakeB	201cbuildB201cbitcoin,B2002B20-30B2.4B1inchB18thB150,000B115B109B106B0627B00edB00a0twoB00a0theeB00a0roB00a0philippineB00a0overB
00a0koreanB00a0internetB	00a0huobiB00a0fromB
00a0bitpayB(zec),B(whichB(th/)B(oc)B(ice)B(gt)B(daa).B(cbi)B(bc)B$9kB$90B$9,000B$8,000B$750B$65B$64,895B$5kB$5.7B$3.3B$3,000B$29B$200,000B$15,000B$130Bzurich-baedByoutube.Byour.orgByen.Bye,BxvgBwu-tangBwrong,BworryingB
wonderful,B	wonderfulBwiredBwinter,BwidowBwide-rangingBwealthfrontBwatonBwater,Bwant.BwannacryBwallet/addreB	volunteerB	volatile.BvladBvermorelB	verkhovnaBvenue.Bvat,BvanityBuzeButility,ButilitieButainabilityBurveillance,BuruguayBurgentlyB
unverifiedB
unlawfullyBunkB
unfamiliarB	unethicalB
undertakenBunacrip,Buit,Buhiwap,Buganda,BuernameBued,Budt.BubwayB	ubmittingB	ubcriber.B
two-factorBtweeted,BtuffBtrump.Btrong.BtriveBtripeBtrickyBtribunalBtrial,B	trategit,Btrack.BtougherBtitled,BtirredBtip.B	timeframeBtime)BtiktokBthread,Bthird-quarterBthetreet.com,BtheoreticallyBthemedBterrorBterminationBterabyteBteenagerBtatuteBtation,Btate-ponoredBtarget,B	tanzanianBtank,BtangibleBtance.Btaiwan,Btaff,BtackingBt-hirtBruling,BrufferBruble.BroterBrioBriing.Breview:BreumingB	reputableBrenBremedyBrelocateBreliabilityBrelentleBreitBreinB	reflectedBreferencingB
referencedB
reearcher.Breearch:BredmanB
recruitingB	recovery,BrecommendingBrecognizableB	received,BrebrandBreapBreaoningBreactBraoulBranom.BrallieBrace.BquireBquickerBquawkBqualifyBquahedBquadrigaBpuleB
propagandaBproof.Bpromie.B
projectionB	programmeBprofitedBprofeional,B
productiveB	proceededBprivacy-orientedBprinceB
prefectureBprediction,BpraieBpracticallyBpound,Bpopulation,BpopularizedB
pompliano,BpoliBpoland,BpoeeB
plummetingBpledgeBplant,BplaneBplainBpitchBphyically-ettledBphillipBphere,BpgpB
permiionedBperiodicBpend,Bpeirce,B
pectacularBpeak.BpawnBpauedBpartialBpark.BparaguayBparBpaloBpairing.BpagnolaBpacex,BpacalBovertakeBoverlyB	overight,BoverhaulBormanBorchetratedBoptBonegoldBon-rampBoliverBoleary,Bokcoin,BokBoffenderB
objective.Boaka,BnyagB
noticeableBnoted:Bnorway.Bnode40BnobleBnirmalaBnew:Bnever-endingBnetherland.BnetflixBnem,B
negativelyBnegaraBnavalBnativelyB	national,BnacarBmvpBmurderBmullingBmucle,BmoverBmotafaBmith,BminkB	minimalitBminiBmineableBmine.B	millionthBmicrofinanceB	metavere,BmenloBmeltdownBmeenger,BmayoralBmature.BmaticBmateringBmateBmatcheBmany.BmanufactureBmanner,B	mandatingBmajetyB	magazine,BmacroeconomicBlunchB
low-incomeBlook.Blogo.BloggedBliving.BlitenBlewiBlevinBleeryBleepingBlbxBlayer-2BlakeBkyrocketBkodakBkicktartBkennethBkennedyB
kazakhtan.BkathleenBk.imB	judiciaryBjudgingB	jpmorgan,BjaxxBjanBjailedBiuance,Bit:BirerBireland,BiphonedBiphoneBiolatedBiohkB	inurance,B	intuitiveB	intriguedBinternationally.BinterferingB	interbankBinteniveBinnovateBinnocentBingledBinformativeB
inexpeniveBindirectB
indicator.Binclude:BinauguratedB
inadequateBimprovement.BimponB	impoible.Bimage.Bii.BignaledBignal.BigB	identifieBhyperinflation.Bhutdown,BhungryBhumbleBhremBhowcaingBhoward,Bhorizon.BhoffmanBhieldBheritageBhead,Bhbc,Bhaven.BharborB
happening,BhankeBhadntBhacking,BgruelingB
groundworkBgox.Bgold-backedBglaBgiftingBgeneration.BgearedBfuionBfrom,Bfriend,BfriedmanBfreedom,BfreedmanBfrance.Bfood,Bfollow,BflyingBfloatedBflexibilityBflexeBfirtly,B	fireblockBfinmanBfilm,BficoBfeedingBfee-freeBfbi,Bfail.Bfahion,BfacetBf2BexploredBexplicitBexoticB	exchangerBeventualBevadingBequoiaBepiode.B	eparationBenviageBenuingBentertainment.B
enterprie,Benglih-languageBengland,Bengineering,BenduringBenduredBendeavorBencodedBempire,BeminarBemerged,B	elliptic,Belling.Bele.BeikonBefficiency,Becure.B	ecretary,Becret.BecalateBeaterBeat,Bearning.Bdubai.B
dubai-baedBdualBdrama.BdragB	draconianBdowntownBdone.Bdone,B
dominance,B
dollar-cotBdoge.B
documentedBdnm.BdivorceBdiverityBdirectorateBdirector-generalB	directingBdipoeBdipeningB	digitizedB	digitallyB
diclaimer:BdiatrouBdevieBdevaluedBdepoitorB	depoitingB
deperatelyBdeliberatingBdeirableBdeepetBdecryptB
day-to-dayBdatabae,Bdalio,BctB
cryptoteelBcryptopunk,BcryptokittieBcryptocurrency-friendlyBcryptocurrency-backedBcrutiny.BcrudeB	crowdale.BcroeBcro-aetBcriptingBcredit,B	coverage,BcourteyBcourt-appointedBcorruption.BcopiedBcooperatingB
coolwalletBconveyedBcontradictoryBcontract-for-differenceBconnection.Bconideration.Bcongre.Bcongre,B	compound,BcompliedBcompliance,Bcompetitor.BcompatibilityBcompany/wallet/infratructureB	commerce.Bcommented-onBcomment,BcomfortB	columnit,BcolonialB
collector.Bco-authoredBcme,BcloetBclintonBclinicBcleverBclearinghoueB	clearanceBcircumtance.Bcircumtance,Bchool,B	cholarhipBcholarBchinoBchenBcheapetBcharitieB
challengerBchainbetB	certaintyBcermakBcentre,Bcare,BcapuleBcalinBcaixinBcahpayB
cah-ettledB	byzantineBbuzzwordBburry,BbureaucracyBbudget.BbuckBbtc-e,BbruelBbrozeBbritBbrick-and-mortarBbribaneBbrekkenBbreedBbreacheBbrandingBbolB	blueprintBblockfi,B
blockchairBblatedBblatantB
bitunivereBbitriverBbitinfochart.comBbitdbBbitcoin-centricBbitbayBbirthdayBbidderBbeyond.Bbeta,B
benefitingB	behavior,BbeatingB	bangaloreBbalance,BbaelBbad,BbacklahBaward.Baward-winningBaving,Bautin,BauredBathvikBatelliteBanypayBantonBanticipatingBanti-corruptionBantanderBanimocaBanimalBandroid,BanarchyB	analytic.Banalyt.BamuelBalready.BallyB	allocatedBallaireBalbeitBakinBakakov,Bak-me-anythingBafter.BafetBafeguardingB	advocate,Badvice.B	advertierBadmiionBacentBabuingBabtractBaaultBa+B97B80,000B7thB500.comB30thB250,000B24-hour,B210,000B2030.B2025.B201ctopB
201ctoday,B201ctillB201ctartingB201cplanB201cnextB201cmyB	201clegalB	201clargeB201cicoB201cgovernmentB	201cenureB201celectronicB201cdontB201ccoinB201cbitcoin:B201camericanB201cafeB2018whoB2000+B19thB1997B1600+B14thB13thB1040B100,000+B00e1,B00a0witzerlandB00a0tezoB00a0reearchB00a0peer-to-peerB00a0icoB00a0gmoB	00a0afterB.B**editorB**B(xmr),B(ubi)B(uaf)B(ip)B(imf),B(forB(fatf).B(edt).B(dex),B(ddo)B(ceza)B(cbn),B(bcc)B(bayc)B(ax)B(aum),B(amf),B$40,000B$4.5B$37B$30,000B$27B$22B$19B$16kB$155B$150,000B$12,000B$1000BzecBzealand,Bzaif,BytematicBymbolicByear-on-yearByamBxlmBwon.Bwomen.BwokeBwojakBwoeBwixBwirecardBwing,BwihingBwidomBwhy?Bwhile.Bwhale,Bwealth,BweakerBwatedBwarnerBwarned.BwarfareBwareBwant,BwaleBvote,B	volodymyrBviualizationB	violence.B	violence,BvincentBvideo-haringBvictoriaBvertcoinBvermontB	verifyingBvenue,BvapBvancouver-baedB	valuable.B
validatingBvagueBuyenBuxBuurpBurge,B	upportiveBupplyingBupheldB
uperviion.BunuuallyB
unofficialB
unintendedBunfavorableBunderminingBuncorrelatedB
unchanged.BunawareBummonedBuicide.Bued.BucceeBucce,BubjectedB	ubidiary.BtylerBtwitBturkBturfBtuition.BtroutnerB	trongholdBtrong,Btron.B
triumphantB
triangularBtrezor.BtrevorBtreatingBtranformativeBtraightforwardBtoughetBtorreBtorchB
tokenizingBtokeninightBtoddBto:BtippedBtimulu.BtighterBtiflingBtifleB	thrillingBth/BteynbergBtetifiedBterminatingBtep.BtendencyBtbBtattooBtaperingBtalledBrwandaBrupee.BruleetBruble,BrouterBroubini,BrollupBrole.B
robinhood,BroachBrigorouBridBricoBrevieBreveringBretrieveBretriction,B	retailer.BrequiiteB
reporting,Breplace-by-feeBrepectively,BreortingBreolvedB	renaianceBrelayBreinhartBreimbureBreddit.B
reboundingBreaureBrealitieBreactedBrangedBrakingBrakeBragingBr3Br&dBquickly,BquetionnaireBquarterbackBquarleB
pychedelicBpundit,Bpump-and-dumpBpullbackBpromptB
promotion,Bprogre,Bprofile,B
proclaimedBproceeding.BprobingBprobe.B	proactiveBpro.B
pro-cryptoBpreyB	preumablyBprerequiiteBpremium,BpredominantlyBprecieBprecededBpre-mineBprayBprankBpouringB	portugueeB	populatedBpoorlyB	ponderingB	pomplianoB	poloniex.Bpoitive,B	pocketnetBplaguingBplagueBpivotingBpiceBphilanthropitBphdBph/B	pewdiepieBpeterenBperu,BperuBpermiionle,BperilBpenion,BpegBpeed,BpayzaBpavingBpatent.Bpart.BpardonBpapaBpaionBpaintedBp2hB	ownerhip:BowedB
overlookedBoutput.BouthinedBoutdoorBouljaBorryBoracle,B
optimitic.Bopportunity.Bonly,Bonchain.B	onboardedB	old-choolBoinbajoBoil.BoffloadBoffline,BoffeneBoccaionallyBnyuBnowhereB
notorioulyB
nonprofit,BnonBnomuraB
nominationBnevada,BnelonBneilBneeded,BnearbyBnear,BmyetherwalletBmurrayB
multipliedBmtr)Bmovie,BmotiveBmotherboardBmorriBmoreheadBmonumentBmonikerB	mongolianBmonetizationBmohamedBmoeBmode.BmockBmixer.B	mitakenlyBminerd,B	miletone,BmikhailB	mid-marchB
mid-april,B	microoft,B
miami-baedBmeyBmeme.BmedicineBmechanicB
meaurementBmclarenBmaxwellBmatrixBmatoniB	maternodeBmarriedBmarkuBmarket.bitcoin.com.Bmargin.Bmap,Bman.B
maintainerBmacBlynchBluringB
lundeberg,BluckilyBlootBlooeBlongtandingBlong-runningBlong-latingBlock-inBlocalbitcoin.comBllp,B
lithuania,BlipBlinux,BlinBlighterBlight.B	liberlandBliabilitie.Blerner,BleoBlender.Bledgerx,BlaughBlatviaBlarge.BlaoBlandedBlaggingB	lackluterBkyivBkyc.BkumarBkudelkiBkremlinBknockBkikB	kidnappedBkickingBki,BkeywordBkeptic,Bkenya.B
kazakhtan,BkanyeBk.BkBjutifyBjutifiedBjr.Bjournal,BjohnnyBjetBjaneBjaimeBiphone,B
invitationB	inventoryB	inventor.BinvadedBinto,BinterwalletBinterferenceB	interfereBinteret-bearingBinteractingB
intangibleBinpectorBinnoilicon,BinfringementB
influence.BinefficientB	indoneia.B	indiegogoBindefinitelyBince,Bimplemented,Bimage,BilenceBiii,BignitedBide,Bichuan.BhudderBhrinkBhotel.BhorribleBhopkinBhope.Bhop.comBhoBhit.Bhit,BhimbunBhiltonBhijackedBhigh-endBhiftedBhetty,BheraldedBhelpfulB	headline,BhatnerBharmfulBhardlyBhard.BhanyeczBhangBhamperBhameleBhaleBha,B	guidance,BguardianBgrimBgreedBgraphB	governor.Bgovernment-iuedBgonnaBgoing.BgloomyBglanode.Bglanode,BghanaianBgeographicalBgenuineBgeneBgemini,BgarnerBgarlinghoue,Bgaming,BgamblerB
fyookball.BfurieBfundamental.BfunctionalitieBfun.B
fulfillingBfrenzy.BfoughtBfortyBfortune.Bformat,B	forbiddenBflow.Bflight,BfleeBfixingB	five-yearBfirt-quarterBfire,BfinmaBfingerB
finder.comBfincen,BfilterBfierceB	fibonacciBfiat-peggedBfiat-cryptoBfed,BfavourBfaucetBfatherBfakedBfailure.Bface.Bfa.Beye.BeychelleBexploitationBexploit.B
explicitlyB
exemption.B	exchange;BexaggeratedBevilBeverything,Beventually,Bevening.Bevader.BevaderBeriouly.Bera,BeponymouBepinozaBeparatedBenvelopeBentrieBentity.B
enthuiaticB
enterprie.BencompaeBembarkedBelling,BellenBelf-certificationBeh/,Begment,B
efficient.BecuadorBecretlyB	ecommerceBebayBebang,Beaier.Bdydx,BdviionBdurationBdrewBdonnellyBdollar-denominatedB	ditinguihB	ditancingB
directive.BdipenaryBdimialBdifference.BdicordBdiaporaBdialogueBdiB	devatatedBdeterB	detentionB
deliberateBdelay,BdeiredBdeigningB	defendingBdefeatBdefault,Bdeciion-makingBdead.Bdb,Bday?Bdappradar.comBdamnBd.c.Bcybercrime.Bculture.Bculture,BcrytalBcryptobuyerBcrypto.com,Bcrypto-aet.Bcript,BcrimeaB	creditor.BcovertlyBcountermeaureBcount.Bcouncil,Bcorruption,BcorridorBcoronaviru.B
cornertoneBcopayBcontructingB
contructedBcontroverial.BcontractingB
continued,B	continue:B	conpiringBconnectivityBconiderably.B
congetion.BconfuedBconequence.Bconenu.BconentB
condition,Bcompromied.B
compoitionB
complementB	commenterBcommemorateBcome,B	combined.BcolumnBcolorfulBcollape.BcoinrailBcoinoBcoinmarketcap.BcoinhiveBcohortBcohenB
cofounder,B	cobinhoodBcoatalBcmaBcloely.B
client-ideBclean,B
citigroup,BcircumtantialBchool.BchiBcheapair.comBchangerBchanged,Bcentury.B	centrallyBcentralization,Bcbdc,B
cautionaryBcatleBcategorizedBcatalogBcartenBcarcelyBcaraboboBcanningBcanada-baedBcaling,BcaaBbv.BbutlingBbullih,BbudBbtc)BbruntBbrown,Bbrook,BbroadenB
broadcaterB
broadcatedBbrave,BbrainBboxerBboutiqueBbottleBboth.BborderleBboom,Bbody,BbloodBbloggerBblockchain.com,BblocBblitzBbizarreBbitnovoB
bitlicene.BbitcoreBbitcoin-backedBbitclubBbit.BbindingBbikeB	benefitedBbegin.BbearerBbeach,BbchnBbaycBbarclay,BbarackB
bandwagon.BbancorBbalancedBbag.BbadlyBbackground.Bbackground,BbackerBbackendBb2c2Bayre,B
available,BautoritBauthorization.BauditorBauditingB	audience,BaudB
attractionB	attendingB
atmophere.BatlanticBatlanta,BatiliBath.BarrivingBarret.Barrangement,BarleyBargentineanBarenalBarahBapprox.B
approachedB	apologizeBapexBaoBanymore.BanybodyB	annually.Banniverary,Bangele-baedBamonB	american,BameerBambaadorBamaBaltucherBalphanumericBalliedBallegation.Balert,BakoinBairplaneBahmedB
afterward,BafoulBaembly.BadvereBadopter,Bactor.BacendingBaccuracyB	accomplihB
acceptableBacce,B	abundanceBabilitieB[it]B	[bitcoin]B92B91B70,000B6:B600,000B5xB50+B450B420B4.8B4.5B35,000B315B3.0B2fxB27-year-oldB23rdB216B21,000B2023.B	201cworldB201cwithoutB201cuingB201ctaxB
201creallyB	201cotherB
201cmakingB201cmadeB
201chighlyB201cheB	201cfullyB201cdogecoinB	201cblockB	201cblackB	201cbeingB201cbankB201canyB201candB2019)B2.0.B190B17thB16,000B154B136B120,000B111B10-20B1.8B0bcdB0644B00faB00ednB00e9,B00a0ukB	00a0trumpB00a0lightningB	00a0hedgeB00a0coincheckB00a0accordingB0.3B0.2B0.01B(xmr).B(vc)B(uahf)B(rbz)B(r)B(ptr)B(nt)B(mlm)B(ma),B(kwh)B(ir),B(edt),B(doge),B(dnm),B(defi).B(cmeB(ce)B(btm)B(bitcoinB(aum).B(ath).B$9.5B$800,000B$75,000B$650B$400,000B$34B$28B$250,000B$2.7B$1mB$15kB$134B$100mB	$100,000.BzurichByoubitByncBympoiumBygniaByellen,Byear-oldB	year-longByang,BxinhuaBwriter,BwreakBworthle.BwornBworkhopBword:BwooriBwitneeBwipeBwinner.Bwilliam,Bwill.Bwife,Bwhy,Bwhitepaper.Bwhite,Bwell-etablihedBweedBwebite:BwebinarB
weaponizedBweakneeBwaxBwaughBwater.Bwall,BwakingB
wahington.BwaabiBvromanBvrBvonBvoltaireBvictorBvictimleBverbalBvera.BventB
vehementlyBvault.BvariedB	variationBvaluingButxo,Butc.B	urroundedBurgencyBurface,BuquidB	upported.BuppedBuplandBuperrareB
unrealizedBunpaidBunknown.B	unfoldingBunexpectedlyBundoBunderwritingBundertandableBundergraduateB	undergoneBunday:B	unchainedBuncenorableBun.B	ulbricht.BuglyBugget,BueleBudc.BudanBubioftBub,B	two-thirdBtwinBtv.Bturkey.BtumblerB	tructuralBtrojanBtrivingBtrippedBtripledBtripleB
triggeringBtrideBtribuneB	treaurie.BtreaurieBtranportB
tranmiion.BtranitioningB	trancriptBtrain.BtrailingBtraded.B
tradeblockB	traction,BtowerBtotal-valueBtop,BtoolkitB
token-baedBtockholmBtitaniumBtimelyBtime:BthumbBthirteenBthickBtheydBtheoritBtheoBtextbookBteting,BtetamentBterritorie.BtemptedBtellar.B
televiion.Btela.BteenB
technokingB	taxpayer.BtatutoryBtating;Btat.B	tandpointB	tandaloneBtaken.Btake,B	tabilizedBtaalBtaakiBt-mobileBrupee,Bruian,Brow.BrokkeBrobinBrobberBriledBrich,BriccardoBrhotokenBrhettBrevolut,BrevivalB
revitalizeBrethinkB	required.Breputation.Brepreentative.B
repondent,BrepectedBrepect,BrenewBremitB
reluctanceBregiter.BrefutedBrefugeB
reflectionB
reflectingBrecoureBrecountB	receptiveBrecepB	real-lifeB
real-etateBreactingBrawBravagedB
ranomware.B
ranomware,BrandalBrainBrafaelBrada,BrabbitBquoB
quarantineB
purportingBpunkB	publihed.BptrB
protectiveBpropelBproof,B
prominenceBpromied.Bpromie,BprofoundBprofile.Bproecution,BproblematicBprobe,Bpro,B	privilegeBprior,B
principle.B
prevalenceB
pretendingBpreppingB	preident.B	preervingBpreent.BpredictablyBpractitionerB
potential,B	pot-ovietB	portrayedBporcheBpooledBpolitician.B
poitioningBpocket.BplungingBplcBplatinumB	planning,Bplanned.Bplanet,B
plaintiff,Bpin,Bpierce,BphiloophicalB
philoopherBperryBperpetratingBperonality,BperitBpercentage.Bpenny.BpenderBpecterBpecifieB
pearheadedBpeaking,BpaueBpatentedB	pari-baedBparcelBparagonBpanelitBpalihapitiyaB
paletinianBpalantirBpalaceBpakitan,BpaintBpagoBpabloB
overnight.BoutfitBourcedB	otherwie,Borigin,B	ordinanceBoptingBoptimizeBopined:B
operating.Bopen,BonlyfanBonecoin,Bonce.B	omething,BolvencyBokiBogB	oftentimeBofiaBoff-rampBof,Boccur.BoccupiedB
obligatoryBoblat,BoapB	notorietyB
noteworthyBnotably,BnorthwetB	nonetheleB
non-cryptoBno,BnitiBnguyen:BnguyenBnfcBnexonBnever-before-eenBnettedBnepalBneighborBnegotiatingBneakBnaimBmyterieBmumbai-baedB
multiplierBmuleBmufgBmpBmowBmoueBmotion,Bmot.Bmonth-over-monthBmonitoring.BmonacoB	momentum,BmohammadBmodeledBmixeBmitchB	mirroringBmiredBmined,BmimicBmillionaire,Bmillion-dollarBmillBmiioneBmid-mayBmicrotranactionB	michigan,Bmetamak.Bmen,BmeiBmega-utilityBmeetup.Bmeetup,BmediterraneanBmediciB	mechanim,B
mechanicalBmeari.ioBmeari,BmchenryB
maximalit,B
matrixportBmathB	material.BmarterBmarket?Bmarket.bitcoin.com,BmarkedlyBmarcuBmap.BmantraB
makerplaceBmaker.BmaireB
maharahtraBmahBlunarBlumenBluggihBlucerneB	loophole.Blonget-runningBlolliBlocalbitcoincah.orgBlitteredB
lithuanianBlitenerBliquidation.Bliquidated.Blink.Blink,BlikeneBlike,Blightly.Blie.Blie,BlicenureBliberty.B
liabilitieBleviedBleeveBlecturerBleap,Blead,BlazaruBlayoffB	lawmaker.B	lawmaker,Blate,Blat.BlarimerB	large-capBlarge,BlandfillB	landcape,BlamauBlahBkyrocketed.Bkyrocketed,BkyleBkyberBkumamotoBkorbitBknowledgeableB
knowledge,Bknow.BknightBkleiman.BkiroboBkipBkiok.Bkill.Bkiev,Bkeepkey,B	kardahianBkarateBkanoonBkaliBjouleBjoanneBjeuBjeopardyBjenniferBjay-zBjamaicaBjakeBix-monthBiuer.B	itharamanB
irrelevantB
irrationalB	inventor,Bintroduced,BintradayB	interval.BinterpretedBinteretinglyBintereting,BinterectionBintentionallyB
intenifiedB
intenifie,B	intantly.BinpectB	inknationBinider,B
inherentlyBinger,B	ingenuityBinfluencer,Bindice.BincrementalB
incentive.BinaneBinaBimportantly,BimportantlyBimplemented.Bimplementation,Bimpact.BimmeredBilliquidBiiiBignal,BifinexB	identity,BictBi]Bi:BhurryB	hurricaneBhungaryBhudonBhudderedBhryvniaBhriBhown,Bhowever:Bhortage.Bhort.B	hopefullyBhomepageBholmBhokinonBhodler,BhitorianBhintingBhiningBhinduBhenzhen,BhenryBhekelBheerBhaulBhat,Bharri,BharonBharder.B
happening.B	haolinfryBhanke,B	hamphire,BhakingB	hahpower,Bguilty,B
guideline,B	guidance.Bguet.BgueingBgueB	guangdongBgrefB	greenhoueB	greenbackBgreece,B	graycale,BgraveBgrandmaBgrammyBglanceBgigawattBgerryB
geographicBgenei,Bgemini.Bgateway.Bgarzik,BgartmanBgardenBgarciaBgameplayBgallupBfungibleBfunctioningBfull.Bfriend.B	friedman,B	frequencyB	frankfurtBfoulBfortune,Bfork:BforgeBforetBforemotB
foreeeableBforbe,B	followed,Bfollow:BfogBflorida-baedBflight.BfitzpatrickBfintech,BfinraBfinihingBfine,B	filmmakerBfihB	fidelity,BfetBferociouBfdaBfaringBfantaticBfangBfanfareB	falteringBfailed.Bfail,Bfacing.BeyingBeye,BexualB	extractedB	extenion.B	extenion,BexpreeBexplanation,Bexpire.BexertingB	exchange?BexcerptBexamBexaggerated.BethfinexB	ethernityBethanBeraingBeraedBequenceBequatingBepochB	eparatingBepaBentrepreneurialB	entirely.BentertainingBentence.Bentence,Bengland.Bengine,B
endowment,BendoringBencryption.BencryptBemulateBemmerBemilioB	embroiledBembarkBelipayBeligmaB	elf-hotedBeized.Beither,Begwit,B
effortlelyBefficacyBedt,Bedition,BedB
ecuadorianBecrow,BeclecticBebay,BeateadBealedBe-walletBdytopianBdupingBdubai,BduBdrug.BdroneBdream.BdrawdownBdragonBdrBdownedBdoomdayBdolphinBdocumentingBdiverificationBditribution.B	dinwiddieBdiginexBdiem,B
dictionaryBdictatorBdiappearanceBdetroitBderrickBderB	depreion.Bdenial-of-erviceBdemonetizationBdemie.BdemianBdeltaB	deliverieBdeleteB
defendant,Bdeciion-baedB	dealerhipBdeadlyBdapp,BdamBdaddyBdabblingB	cutodian.BcurtainBcurioBcultBculpritBcryptowatchBcryptonize.itBcrypto-miningBcrypto-fiatBcrutinizingBcrowdfunding,B	crowdfundBcroroadBcredit.Bcreativity.Bcrazy,Bcraze.BcouponBcounterparty,Bcotten,BcotlandBcorrection,Bcoronaviru-ledBcopycatBcoolingBcoo,Bcontruction,Bcontrovery,B	contraintBcontinuationBcontinuallyB	contetantBcontendB
conpiracy,B	congremenB
confirmed,Bconficated.Bconent.BconditionalB
concluion.B	concertedBconcertBconcentrateBconcededBcompromied,B
component.B
completed.B
completed,BcomplementedB
complaint,BcompetitiveneBcompenation.Bcommon.Bcommiion-freeB	commandedBcoinmarketcap.comBcoinkiteBcoindahBcoincidentlyB
co-creatorBcnyB
cloud-baedB
cloed-doorBclayton,BclawB	claifyingBcircumventingBchnabelB
checkpointBcheckedB
character,BcfBcexBceremonyBcereBcenureBcentralized,Bcenario,BceliuBceationBceBcbdBcazeBcateringB	category,BcartoonBcarterBcammyBcambodiaBcaldwellBcajeeBcaino.Bc++BbyrneBbv,Bbuyer,Bbullih.B	building,Bbuffett,BbuffetBbuda,BbudaBbu,BbrownBbroweBbrightBbrettB	breachingBbranche.BbrailBbracingBboxeBbox.BboureBbottom.Bboot.BbogBbleedB	blackmailBblacklitBbittrex.BbittBbitprimB	bitflyer.Bbitfarm,Bbitconnect,Bbitcoin-themedBbitcoin)Bbip91BbionBbetween.Bbermuda,BbenchmarkingBbelovedB	believed,BbelfricBbeijing.BbeholderB
beginning.Bbch/udBbch)BbbvaBbay,BbatmBbareBbanned.B	balloonedBawarene.BavoidedB
automobileBauthoritarianBaudioBauBattetB
attendanceBattempt.B	attacker,B	atonihingBatoB	atmophereBartwork,B	arringtonBaronB	armtrong,B	argument,BarabiaBapyBappraialBappointB
apologizedBapenBaociate.B	anything,Banyone.Bantpool,BantonyB
antiquatedBantimentBantaBangele,B	anarchit,BambitionBamateurBalternatively,BalpineBallegation,BalinaB	alexandreBalarmedBakeBakariBair.Bair,BaidingBaggregator,B
afternoon,BaforementionedBaffirmedB	affidavitB
afekeepingBadvice,B	adventureB	adminiterBadjutedBadalendBachievement.Bach.B
accuratelyBaccountabilityBaborbBaaBa1246B`theB9:00B8thB850,000B6thB5.3B401(k)B3iqB25,000B220B210B201cthroughB201ctateB201crikB201creal-lifeB201cpromoteB201cpotentialB
201cpecialB201coverwhelmingB
201cminingB201clikeB201cletB201clatB201cjohnB201cincreaingB201cinceB201chugeB201cgoldB201cgoB201cfreeB
201cexpertB201ceverythingB
201ceveralB201ccoinbaeB201cbetB
201cbecaueB
201caroundB	201callowB2019?B2018hodlB2018cryptomatoeB2002,B2001B2)B1t.B1999B15th,B145B144B1-2B0xB00fcnB00e3oB00a0tudyB00a0tranactionB00a0revealedB00a0reportedB00a0regulationB
00a0peopleB00a0ixB
00a0iraeliB00a0ilkB00a0highB00a0hahB
00a0googleB00a0federalB
00a0europeB00a0atB**thiB(xrp).B(worthB(vap)B(upto)B(udc)B(tud)B(to)B(lp).B(ln).B(ir).B(ifp).B(ice),B(fiu)B	(fintech)B(eu)B(etc),B(eh/),B(dot)B(doj),B(doge).B(ct)B(cbr)B(c)B(bp),B(bot)B(boe)B(bnb)B(atm).B(atB(aloB(almotB$7.5B$64kB$530B$4.6B$3,800B$3,600B$270B$2.25B$14,000B$13kB$10,000.B$1.8B$1.25B#builtwithbitcoinBzurich,BzhejiangBzhangBzeroeBzec,ByukoByourelf.Byourelf,B	your.org,Byoung.Byou?ByndicateBymbol.ByieldingByear-to-dateB	year-end.B	yamaguchiBxxBxt,BxiongBwrappingB	worldcoinBworld-renownedB	workforceBwork?Bwon,BwizecBwireleBwipingBwindingBwin.BwillyBwihedBwideningBwidenBwi-fiBwhole,Bwho,Bwhen,BwexBweworkBwellingB	week-longBweden.BweaketBwbdappBwave,BwattBwatkinBwaterhedBwarriorBwarrant.BwallexBwalkingBwageringBvulnerabilitie,Bvpn,BvorickBviual,Bviru.Bvinnik.BvikingBviion,BvigilantBvice-chairmanBvia.Bverge,Bverde,BveratileB	vegetableBvault,Bvalue-addedBvalidityB	validatedBv20Butc,BurvivingBurvive,Burveillance.Burprie.BurbtcBupremacyBuppliedBupplementaryBuperviedBupbit.BunwarrantedBunureBunpredictableB
unorthodoxBunlockedBunjutBunivere.BunhappyBunfoldedB	underwentB	underway,B	undercoreBunderbankedBunderageBunconventionalBuncontitutional.BuncommonBultraBuexBubtrateBubidieBubcontractorB
typically,Btype.Btwo-weekB	tuttgart,Btud,Btrutee,Btrouble,Btronger.BtrikingBtreingBtreamerBtraviBtravelbybitB	trategie,BtrapBtranferred.Btranaction)BtrainedBtraded,BtoyBtoutingBtorie,Btorage:Btor,Btone,BtmBtie-upBticker.Bticker,B	thriving.B
three-yearBthree-monthBthoroughB	thorchainBthink,BtheiB
theatricalBtheatreBtetifyBterryBterra,B
terminatedBtemplateBtellar,BteleviedBteelB
technique.Btax-relatedBtavanir,BtauruBtation.Btate-of-the-artBtate-charteredBtatartanBtat,BtaperBtampaBtampBtallyingBtallBtalent.Btaking.BtahiniBtagBtafferBtabletBt17BrunawayBruling.BruhedBrugbyBrtxBroutineBroundedBroom77BrogoffBroenteinBrockedBrobotBrobbedBroadmap,BrikbankBright?BrifeBrifBriddledBricardoB	revolvingBrevokeBrevivedBreveral.B	revealed:BretructuringB
retrainingBrepo.B
repercuionBreorganization.BrentedB	renminbi.B
rememberedB
remarkablyBreleaed,BrelaxBrelaunchBrelationhip.Brelationhip,B	reitance,BreidB	regitrantBregiter,Bregard,B	regainingBrefrainB
referendumBreervedBreelB	reearchedB
redemptionB	redeignedB
redeemableBred.Bred,B	recorded.B
reckoning,B	received.BrealmxB	realizingBreader.Brbi,Brapper,BraeeBradio,BquiznoBqueueBquetionableBqueriedBquellBquebec,Bquadrigacx,BqartBq2,Bputin,BpuruedB	publicly.BptiBprovide.B
propertie.Bprogrammer.BprogrammableBprofitability.BprofeionallyB
proecutor,Bprivacy-preervingB	preventedBpreuringBpreure,BpreidingBprecludeB
precautionBpre:Bpre-regitrationB
pre-launchBpourB	potponingBport,Bpop.BpooretBponor,B
pokeperon,BpointleB
plummeted.Bplot,Bpimco,BpilotingBpiBphotographerBphoenixB
philoophy,BpfBpeudo-anonymouB	petition,BpermiiveBpentagonBpennylvania,Bpend.BpellingBpectacleBpearheadBpeaker,Bpeace,BpdaxBpaywallBpaxo,Bparole.BparoleBparliamentarianBparerB
pancakewapB
panamanianB	palladiumBpailBpae,Bpa.Bpa,BoxfordB
overturnedB	overtakenBoverflowBoutreachBoutlook.BoutlawedBouthwetBouthineBoutgoingBoutedBoutdatedBoutage.BoptionalBoptimim.Bopt-inB	opendime,Bontario.BonrampBonlookerBonlaughtBonionB	ongwriterBone:B
one-to-oneB	one-thirdBominouB	omething.Bome.BolitudeB
offloadingBockBocioB
occurrenceBocB	obituarieBobeionBnydfBnycBnvidia,B	numimaticBnowaday,Bnoting:BnoticingBnortonBnon-violentBnon-governmentalBnon-bankBnomineeBnominalBnikolayB	nightmareB	nightclubBnigeria.B	nicknamedB	newpaper.BnewfoundBnewbieBnet,Bneo,B	negative.BnederlandcheBnearedBnatchingB
narrative.BnappedBnaomiBnano,BmurphyBmulti-billion-dollarBmuch-neededBmuch,BmtfBmoved.Bmotion.Bmortgage-backedB	mortgage,BmorrionBmoothly,BmoothlyBmoon.Bmonthly.Bmom,BmodularBmobile,BmitubihiB
mitertangoBmitake.B
minimizingBmine,BmiltonBmillennial,Bmiing.BmidwetB
middle-claB	miconductBmichelleBmethodologyBmember:Bmember-tateB	meltdown.BmeganBmeddlingBmeconBmean.Bmeaging,BmaximBmawonBmavBmauricioBmaturedBmature,B
maturationBmathematicianB	matercoinBmarylandBmaryBmartetBmarketwatchBmappedBmanuelBmambafxBmagicalB	magazine.B
maachuett,Blump,BlumberBlullBludwigBlozanoBloyalBlovenia.Blovenia,BlouiianaBlookoutBlooelyBlongtimeBlondon-headquarteredBlocalcryptoBlittle-knownBlimBlightingBliechtentein,BlibelBliarBlennon,Blegilature,Blechter.B	leadblockBlbryBlayingBlayer.BlaxBlavihBlauraBlarvaBlarBlandingBlammedBlamBlaelBlackingBlackedBkucoin,B	kritalinaBkraken.BkookminBkomodoB	kommerantBknown.BknockingBking.Bkind.Bkid.BketchyBkaplanB	jutifyingBjutificationBjuliuBjudgedBjournal.BjoongangBjoke,Bjohnon,BjohanneburgBjmpBjikhBjijiBjerey.Bjerey,Bjapanee,Bjackon,BixtyBix,Biued,BitanbulBitaBiphonBipfBio,BinvokingB	invetableBinvaiveBinu,B	interveneBintermediateBinterfaxBintenifyingB
integrity,BinolventBinight,Bingle,BinghBinformationalBinflateB
infightingB	infectionBindoorBindefinitely.Bindeed.B	increaed.BincompatibleBincentivizingBinaugurationB	in-walletBimprovement,B
important.Bimport.B	impoitionBimply,B	impactingBimmeneBilB	ignoranceBidolBideologicalBid.Bi)BhypotheticalBhunBhuman-readableBhugheBhudBhuaweiBhoveredBhortcutB	honetcoinBhockwaveBhobbyitBhindiBhillerBhigh-net-worthBhigh-frequencyBhift,Bhi-techBherrodB	hermitageBhermanBherelfBherald,Bhenzhen-baedB
helicopterBheitantBhederaBheart.Bheadquarter,B	headline.BhatteredBharrionBharmedBhangingBhandingBhandhakeB	hamphire.BhakeBhailBhahedBh.BguoBgundlachBgroupedBgroup)B	groundingBground,BgripBgregoryBgreenerBgrant,BgrahamBgraffitiBgovernment-inducedBgoodbyeBgolangBgigabyteBgigaBgiftedB
gibraltar.Bgiant.B
giancarlo,B
geothermalB	georgievaBgeniuBgeforceBgbtc,Bgate,Bgap.Bgaming.BgamificationB	gambling,Bg4Bfuture?BfurnihBfundamentallyBfun,BfullerBfull-fledgedBfud,Bfrozen.BfringeBfreemanBfreeingBfree-marketBfrayBfraudulent.Bfourth-largetBfound,Bforward-thinkingBforgingBforecat.BfootholdBfondB
following.B	fluctuateBflippingBflippedBflaw.BflaggedBfire.BfinetBfilBfifaBficherBfew.B	ferventlyB	feedback.BfearingBfavoringBfaultyBfatet-growingBfamilie.BfameBfallacyB	falkvingeBfailed,B
fahionableBfacebook-backedBezB	expanion.BexecB	excludingBexchange.bitcoin.comB	exchange)BexceptionallyB	evidence,Beuropol,BetupBetchedBerodeBerixBergeiBera.Bepiode,BeotericBeo.Bentry.Bentrepreneur.B
entrenchedBenlitedB	engineer.Benegalee-americanBending,Bencryption,B	encounterB
emphaized.B
emphaized,BeminBemailedBelixxirBeliminatingBelfieBelfB	elewhere,BelegantBelectricity,BeenceBedition.Becrow.Becret,B
econd-handBecobarB	eateadingBearth.Bearning,BeaonedBeaily.BeaietBe-11Be,B	dwindlingBdropped,Bdrama,BdprkB	downtrendBdowntimeBdominoBdoing.BdoggedBdoe.Bdoe,BdocuerieBdmm.comBdmgBdiviion.BdiviibleB	diruptingBdiruptedB
directive,BdinnerBdimiingBdimied.BdimieBdiguiedBdigitalxB
difficult,B	dicourageBdiapprovingBdiappeared.Bdiamond,BdfBdex.topBdeveloper.bitcoin.comBdevelop,Bdetination.Bderibit,BdeplatformingBdentalBdentB	delivery.Bdelhi,B	delegatedBdegenereBdefinition.Bdefault.BdeemBdecentraliedBdebtorBdcgBdc,BdarrenBdapp.Bd.BcyworldBcypherpunk,Bcutting-edgeBcutomizableB	cutodian,BcurtiBcurrency-baedBcurioityBcureBcto,BcryptonightBcrypto:Bcrypto-poweredBcrypto-infuedBcrypto-enthuiatBcrypto-aet,BcrunchBcruhingB
criticallyBcriminalizeBcrii:BcreeningBcreatureB	coworkingB	courthoueB	countrie:Bcounterpart.B	counteredBcount,Bcot-effectiveB	correctedBcorrect,Bcorner.BcooperBconvention.BconumingBcontructiveBcontruction.B	contrary,BcontractualB
contractedBcontract-enabledBcontinentalBconitedBconiderably,BcongerB	confuion.B
confrontedBconfirmation.B	condemnedB	concealedB
computing.B	complicitBcomplicated.BcomplainBcompactBcommunitie.Bcommunitie,B
commodity.Bcommiioner.Bcomment.BcommencementBcomfortablyBcomedyB	colorado,B	colombia.BcolinB	coldcard,BcoinbankBcoderBcodebae,BcoboBcluterBclueleB
cloudflareBcloe,BclipBclearnetB	cleanparkBclayton.Bciphertrace,BchuggingBchuckBchoice,B
chinee-runBchief,BchaumBchatter,B
chainlink,BcfaBcent.Bcenario.Bceae-and-deitBcboe,Bcaue,BcathingBcat.Bcareer,Bcare.Bcardona,B
caramucci,BcaptainBcapitolBcapitalizingBcandal,Bcan)BcalmBcalability,B	cahfuion,BcahcriptBcah-baedBbyte,Bbureau.BbulletinBbullardBbudget,BbuddingBbtg,BbrooklynB
brokerage,BbroadlyBbridge.BbrennaBbreezeBbread,Bbreach,BbranonBbranche,Bbranch.BbrainardBbraceBbox,Bboton,BbotherBborn.BboottrapBboomerBboneBbombB	bollingerBbojBbogotBbodenBblog.BblockerBblockchain-focuedBbleakBblackoutBbitquickBbitkan,B	bitgrail,BbiteBbitdeerBbitcoin-peggedBbitcoin-landBbitcoin-acceptingB	birthday,BbillyBbigwigBbiden,BbiancoBbfxBbeware!BbeverlyB	berluconiBberlin-baedBberlin,B
benchmark,BbeltBbelgium,Bbelaru.Bbeing,Bbeijing-headquarteredBbegun.B
beginning,Bbegin,BbegBbeeple,Bbecaue,Bbchd,B
bch-fueledBbarrBbargainBbannerBbanned,Bbankman-fried,BbanditBbaldwinBbailedBbafin,Baylor,BaverB
automationB	authenticBauterityBauieBaudit.Baudit,B	atoundingBarticulatedBarrangedBaroraBariingB	argument.Barena,B	architectBarabianBapprehendedBapplicability,BapiringBaortmentBaortedBanxietyBantonioBanti-bitcoinB	antander,BannumB
annualizedBanderonBandbox,BanarchapulcoBamung,BamplifyBaml/cftBamexBamaingBamahB	alternateB	alt-rightBalong,BallieBallenB
allegedly,BalipayBalike,BalgorithmicallyBalgorandBalexeyBalexaBalecBalcorBalamedaBaked:Baked,BaitingBairport,BairbnbBaide,BagorimBaement,BaeedB
advantage.Badvance.BadaptingBadageBacronymB	acramentoB
acquaintedBackermanBacertainBaccrueB	acclaimedBabue.Babue,BabideBabhaB
abandoningB@B9:30B9.5B9,500B7000B7,500B535B500.B42,B4.3B4,500B37-year-oldB37,000B365B33,B32mb.B3.8B3,500B2ndB2faB29thB28thB27thB24/7B230B20xB2026theB201cyeB201cwortB201cworkingB201cworkB
201cvaneckB201cueB201cu..B201ctranparencyB201ctenB201ctablecoinB201crichB201cprivateB201cpaymentB201copenB	201cofferB201cnothingB201cnftB201cnegativeB201cmartB201clotB201cilkB201chodlB201cgoodB	201cfraudB201cevenB
201ceekingB201ceconomicB	201cearlyB201cdr.B201cditributedB201cdataB201ccryptoaetB201cbinanceB201canotherB	201caboutB201c2020B201c100B2018newB2018inevitableB2018iB2018highB2018goldB2018decentralizedB2018communityB
2018attackB2018aicB2000.B2-3B19jB1987B107B104B100.B1.6B1,800B1,600+B1,400B1,100B	1,000,000B0x,B00edvarB00e9nB00a0youB	00a0worldB00a0viaB00a0ueB	00a0theirB
00a0tetherB00a0preparingB00a0paulB00a0openbazaarB	00a0largeB
00a0globalB
00a0frenchB00a0extremeB00a0ethereumB00a0decentralizedB
00a0circleB00a0cftcB00a0canadianB00a0bittrexB00a0bitcoin-baedB	00a0atohiB00a0anotherB0,B***B(ud)B(ticker:B(tae:B(racib)B(pboc).B(nyag)B(nfl)B(mti),B(mfa)B(mb)B(ipo).B(iamai),B(hmrc)B(gud)B(gme)B(fma)B(finma).B(cbeci)B(bp)B(bnb),B(bip)B(ato)B(api)B(2fx)B$71B$6000B$55kB$5.1B$47B$4.3B$4.2B$380B$300,000B$240B$232B$20,000.B$2.9B$19kB$170mB$165B$13,000B$11kB$11,000B$10,000,B$1,500B$1,000,B	zimbabwe.BzietzkeBzencahByuzoByuriByuedongByrianB	youtuber,ByoniByonhap,Byear?ByaleBxthinnerBxrbBxemBxapo,BwychB
wrongdoingBwreakedBwrathBworthyBworkflowBwoorannaBwoman,BwolfBwilon,BwildlifeBwildfireBwi-baedBwhitleblower,BwhereinBwherebyB
whereaboutBwheelB	whatoeverBwhatapp,BwhartonBwhale.BweighingBweekly:BwebmaterBweapon,Bwbtc,BwbBway:Bwave.BwatcherB	watchdog.Bwarned,BwarmedBwaning.BwallaceBwaitlitBwage.Bwa.BvyingBvulnerability.Bvulnerability,BvowingBvoidBviuallyB
violation,B
vihwanath,B	vihwanathBvettedBvetoBverigeBverification,Bvenezuelan.B
vancouver,B	valuable,BvalerieBvacancieBv1B
uzbekitan,ButmotBurveyingBurl.BupermajorityBuperintendenceBupercomputerBupect.BunutainableBuntraceableBunretrictedBunprofitableB
unpendableB	unlockingBuniwap.BunhineBungBunfoldBunemployment,B
undetectedB	underway.BundertandablyB
undertand.BundergirdingB
underervedB	uncenoredB	unauditedBunacrip.BummarizeBultra-ecureBuie.BufjBufficientlyBuex,Buele.Bue-caeBudan.Bucceful.BucBuburbBubtanceBubtackBubetBuber,Bubcription.BuahfBuage,Bua,Btype,Btwitch,Btv,BtutorialBturmoil.BtunnedBtrzezczkowkiBtrutingBtruth,B	tructure,BtruckingBtroitkyBtretcheB	trending.B	tranpiredBtranparent,BtrannationalBtranformation,Btrajectory.Btraight.Btraditionally,BtradingviewBtracker.Btracker,BtraceabilityBtpBtorylineBtoriedBtoppleBtongueB	tomorrow.B	toleranceBtoken)BtoatBtitle,BtightlyBtiffBthrough.BthroneB
thoughtfulBthorntonBthinkerBtheorie.B	teynberg,B	terrorim.BternoaB	tentativeBtenneee,BtenfoldBtenantBtemperatureB
televiion,BtelekomB	telegraphBteerB
techcrunchBteady,BtbtcBtayyipBtay-at-homeBtaxed,BtavanirBtatitaBtate-operatedB	tarvationBtarbuckBtar.BtapleBtankedBtandemBtamperedBtaiwan.BtaipeiBtah.Btaff.BtabilizeBtaaki,Bt-hirt,Brunning.BruinedBrtgBroyaltieBroyBroute.Broundup.B	roulette,BrouletteBrollercoaterBroiBroger,BrodBrocoBrockyBroboticBrobinonBrobertoBroaringBroadhowBrk,B
ringleaderBridiculoulyBrichoBricherBrialBreveal,BreuableBretracedBretoringBretitution.BretainedBretail,B	repurchaeB	reported:BreplyingB	replicateBrepairBreortedBreort,B	reopeningBreolved.Brent,BrenminB	renderingB	remindingBremain.Bremain,B
relocatingBrelief,B	rejected.BreidenceBrehapeB
regulator:BregreionBregime,BrefugeeBrefrehB
reervationBreelingBredefineBrecord-highBrecipeB	receptionBreceipt.BreauringBrealtorBrealm.Brealm,B
real-time.Bready.Breading.Breach.B
rbi-backedBrbfBrare,BraptureBrank,BrailBraied.Braied,Bradar.Bradar,BrackBr2Bquote,B	quotationBquoine,Bquarantine,B
qualifyingBq3,Bpyramid,BputnikBpuruitBpurelyBpure.io,B	purchaed,BpurBpunjabBpump.Bpump,B
prudentialBprovocativeB	proviion,BproudlyBprotet.BproppedB	properly.B	properityBproper.B
propellingBpropagatingBpronouncementBproliferateBprogrammer,B
programmedBprofeionB	proecutedBproductivityBprocureBproceeding,Bproceed,B
prioritie.BprionerBpring.Bprime.B
prevalent.BpreuredBpreumedBpretendB	prematureBpreident-electB
preferringBpreferentialB
predictiveB
predeceor.B
predeceor,B	precariouB	preadheetBpre-ipoBpre-forkBprawlingBpraadBpow.BpouredBpound.Bpot-convictionBportraitBportableBpornhub,B	popularlyBpopecuBpodcaterBpodcat.BpnetworkB	plentifulBpledB	platform:B	planning.BpitchedBpinpointBpilot,BpillBpilingBpicture.Bphrae,BpharmacyB	petition.BpeterffyBpetahahB	pertinentBperpetratorBperonal,BperonaBpennieBpennedBpen.Bpeg.Bpeer,BpecifyBpecification,Bpaul.BpauingBparticipation.Bparticipating.Bparticipated.Bpari.Bpari,B	paradigm.BparadieBpaport.BpaphraeBpanic,BpakitaniBpaid.Bpaed,B	packagingB	ownerhip,Bovertock.comBoverlordBoverhadowedBovereenB	outrageouB
outourcingBoutourceBoutider,Bother?BotcqxBort.BorionxBoriginatingBorganicBorbitBorareBophiticated,Boperational,BopenwapBopenneBopendimeBopenbazaar,B
open-endedBopcode,BopB
ongwriter,Bongoing.BonerouBonecoin.Bometime,BomebodyBoligarchB	olidifiedBolgaBoklahomaBohio,Boh,Bofficially,Boffene.B	offendingB
off-chain,Bobama,Bob1,Bnydig,BnubankBntBnowaday.BnoviBnoveltyBnovaBnotchBnot-for-profitBnorilkBnon-bindingBnomadBnoelBnoahBnineteenBnikolaoBnike,Bnight.BnigelBngoBnft-baedBnext-genBnetflix,Bnet.B
negligibleBneelBnearetBnchain.B	nathanielBnathanB
narrative,BnapBnamibiaBmviBmurder-for-hireBmunichBmulvaneyB	multitateBmultiplyBmultibitBmugglingBmoothly.BmoonbeamBmoon,B	montreal,BmontrealB	monopolieBmonoonB	mongolia.B	mongolia,BmolyneuxBmodifyBmodificationBmoderateBmocow.BmobBmoaBmmB
mitigatingB
minneapoliBminitry.BmindfulB	mimickingBmillionaire.B	million);Bmiller,B
mileading.BmilanBmie,Bmid-novemberB
mid-march,Bmid-december,Bmid-decemberB	mid-aprilBmicrobloggingBmiconceptionBmickBmichealBmiappropriatingB	meticulouB	metavere.Bmetamak,Bmet.BmercBmentalB	memo.cah,B
memberhip.BmeltingBmellon,BmehB	megawatt.Bmedium.Bmedium-termBmecumBmeatBmearB	meaningleBmckineyB	mcdonnellB	mcdermottBmccourtBmaxineBmathewBmaterializedB
matercard.BmarttiBmartbtcBmarket;Bmarket:Bmargin,Bmanaged.BmalteeBmalleabilityBmakedBmake.Bmaduro.BmadneBm30++Bm&aBlyingBlowdownBloverBlordB	longevityBlonger-termBloneBlogicalBlocateBlocal.bitcoin.com.Blocal.bitcoin.com,BloadableB	livetreamBlited,B	liquidateB	lingeringBlinedBlimited-editionBlightweightB
lightning,B	lighthoueBlight,Blifted.B	lifetime.B	lifetime,B	licening.B	licening,BliceBlibrary,Blibra.Bliberty,B	leverage,BleveledBletter.Blender,B
legitimizeBleepBlebedBleaeBlawyer.Blaw;B
launchpad,BlaudedBlatvianBlatam,Blaptop,BlapBladyB
laboratoryBl.B
kyrgyztan,B	kyrgyztanBkutcherBkurtBkurodaBkryptoinBkryptoBkokehBkoinexBknoxBknow-howBkiok,BkingpinBkill,B
kidnappingB	kidnapperB
kicktarterB	keychainxBketchBkeplerkBkelly,BkatieBkathrynBkaperkyBkano,BjuventuBjutice.B	judgment,B	judgementBjudge.Bjr.,Bjourney.BjokerBjoke.BjoeyBjianguBjay-z,BjavierBjavaBjaniceBjaitleyBjail,BivyBiued.Biuance.BiriB
irael-baedBio.Binvetigator,BinufficientBintrumentalBintruionBinto.BintimateB	intigatedBintereting.BintenifyBintelligence.B	intantly,B	intalled.BintahiftB	intagram,BinroadBinhabitBingh,BingeniouBinfluencer.BinfiniteBinfectBinexperiencedB
inequalityBineffectiveBindice,B
indicator,B
indicativeBindependent,B	incurringBincurB	incrementB
increaing.BincorporationB	incluion.B	in-browerBimulatedB
important,Bimportance.B	impoible,Bimplication,BimpletB
imperonateB
impairmentBimf,BimaginedBimaginationBilentBignoringBignore.Bignatov,Bifinex,BierraBidgBideallyBickBhyteriaB
hyperblockBhydrawapBhurunBhumorBhugoBhuffle,BhouemateB	houehold,BhouedBhorterBhornetBhordeB
hopitalityB	hopitableBhoodBholiday,Bhokinon,BhogeBhmrcBhmBhitcoinB
high-yieldB	high-techBhiccupBhettyBhercegBhelixBhecticBhebeiBhealth,Bhead.Bhe/heBhaye,BhawnBhavkatBhave.BharveyBhareholder.Bhared:Bhardhip.Bhard,BharareBhappy.BhanoiBhake-upBhailingBhahdexBha.BhBguru,BguitaritBguindoBguieBguiBgrid,B
greenwood,BgreenpanBgreen.B	grapplingBgrapheneBgradualBgoxxBgoodmanB
goldentreeBgold?Bgod,BgoatBglyphBglobal.BglitchBglennBgilbertBgift.BgifBgieleB
gibraltar,Bghana,Bget-rich-quickBgenericBgenderBgdax.BgazetteBgaugingBgartnerBgarcia,Bgamer.B	gambling.B	galvanizeBgadgetBgabbanaBgabBga,Bg.BfuzexBfund:Bfull-erviceB	fulfilledBftcB
frutrationBfruitBfrontierB
frictionleBfrictionB
freero.orgBfreerBfour,BfountainheadBfoteringBfortunately,B	fortnightBforeeeB
footprint.BfootingB	followed.BfmaBfmBfluctuation.Bflow,Bflaw,BflareBfit.B	firt-timeBfioBfiniteB
financial.B	filecoin,Bfile-haringB	fiduciaryBfetival.Bfetival,BfetchBferventBfelixBfelicaBfeiBfee:Bfee!Bfear.BfcoinBfavor,Bfather,Bfater,Bfarming,Bfarmer.BfarleyBfareBfanduelBfan,Bfailure,Bfad.BfacultyBfactory.Bf.B
extractingB	extortingBexploit,B	explodingB	explainerB
expirationBexperiment,B	expanion,B	exhautiveBexercie.B
exchanged,Bexchange-ecrowedBexchange-baedBexahah,B
evolution,BevokeBeventh-largetB	even-timeB
evangelit,BethoBeth-baedBetadoB
etablihed.BerthaBerror,BerodingBerbiaB	equation.BequationBequateBeoul.B
enviioningBenticingBenticedBenterpriingBenrichBenoriumBenoB	enigmaticBenibleBenglih,B	endeavor.BencyclopediaB	empoweredB	employer.B	empiricalBemmanuelBemitBemirate,BeminemBelloff.BelkiBelite,BeliminationBeligibilityB	elf-tyledBelf-regulation.B
elf-cutodyBelevateBelectronic,B
elaborate:Beizure.Beip-1559Begwit2x.Begwit2x,BegregiouB
efficient,BefccBeen,BeditedBedaBecurity-baedBection.B	ecretary.Becond-quarterB	econd-motBeckBecb,BeatingBearn.comB
early-tageBeae.Beach,Bdut.Bdublin,BdtcBdrwBdrive.Bdreden,Bdramatically.B
dragonmintB	dragonflyBdraggingB	downward.B	downizingB
downgradedB	downgradeBdownfallBdoubt,BdoublerBdouble-lifeBdoom,Bdong,BdometicallyBdollar)BdolceBdnyBdnaBdivulgedBdivulgeBdivingB
divergenceB
ditinctionBditchingBditantB
diruption.B
diruption,BdireBdipute.Bdipute,BdiplacedB	dipenarieBdilemmaBdiguieBdigitizeBdigitizationBdiggingBdifferentlyBdieaeBdie,Bdid,Bdicued,BdictateB	dicovery.B	dicloure.Bdicloed.BdickBdiater,BdiarioBdiagree.BdhBdevoteB
deviantartBdevaluation.B	detroyingBdeterioratingBdepriveBdepreingBdepreciatingBdepotBdepictB
dependenceBdeneBdemocratizeB
democracieBdemiBdeltecB	delivery,B
delinquentB	delightedBdeliberationBdeleted.B
deignationBdeignateBdefillama.comBdefB	declared.BdecentralizingB	decentralBdearBdeanBday:BdauntingBdata-carrier-izeBdarkideB	dark.failBdalla,BdabbleBcyrulnikBcypherpunk.BcvBcutomaryB
currencie:BcuringB
cumberlandBcryptoruble,Bcryptokittie,Bcryptographer,Bcryptocompare,Bcrypto-communityBcrowd.BcropB	criticim,Bcritic,B	cripplingBcreepingBcreen.B	creditor,BcredibleBcred,BcrappyBcrap.BcoxBcowenB	coverage.B	courtroomBcottihBcotanzoBcorroboratedBcorrection.Bcoronaviru-drivenBcoringBcoredB
cooperatedBcontroverieBcontradictingBcontractionBcontetedB	containerBconider.Bconglomerate,BcongetedBconfineBconfigurationB	conciergeB
compoundedB
component,BcompoiteB
competenceB
compenatedBcommunicatedB
committee:Bcommitment.BcomebackBcomB	colorado.BcollinBcollape,B	coinfloorBcoinflexBcoindcxBcoercionBcodedBcocaineBco,BclyntonBcloely,Bcloed,Bcloe.B	climacticB
clientele.B	clickbaitBclick.B	clevelandBclearerBclarity,Bclaimed,BclaifieBclahBclae.B	citizenryBcitibankBcitiBchoruB	children.BchengduBchefB	cheduled.BchattingB	charitie.B	changellyBchair,BcezaBcentralization.B	cenorhip,BcementedBcctvBcbn,BcbiBcbeciBcavalierB	cautioulyBcaryBcaruoBcarlyleBcareer.Bcarcity.Bcarce.Bcarbon-neutralB
capitalit,B	capacity,BcannedBcandle,BcancelBcamp,BcammingBcammed,B	cambodia,BcalibraBcalendarBcalculatingB
cahhuffle,Bcafe,BcacheBcacBbyte.Bbutton,Bbuterin.BburtingBburtBburryBburritoBbundleBbullard,Bbull-runBbull,B	buineman,B	building.BbuietBbudweierBbud,Bbtcparer.comBbtc2Bbtc.top,Bbtc-e.Bbrother,Bbroke,B	broadcat,B	brilliantBbrighterBbrief:BbrewdogBbrevanBbrendanBbraunB	brand-newBbranch,BbraggingBbrady,BboyartB	boutique,B	boundarieBbought,B
bottleneckB
borrowing,BboothBbooking,Bbomb,BbogotaB	bogdanoffBboeBbody.BbluzelleBblockvetBblocktream,B
blocktowerBblockchair,B	block.oneBblenderBblendBbleingB	blankfeinBblanketBblame.Bbitwage,BbittreamBbittopBbitmex.B
bitlicene,BbitgetBbitfury,B	bitfract,Bbitcoincah.orgBbitcoin-onlyBbitargBbit.comBbirthedBbirkBbillionaire.BbeowulfBbenefit,B
beltracchiBbeaconBbch-acceptingBbcahBbattlegroundBbarwickiBbartlettBbarrageBbarbaraBbank:Bbank)BbananaBballotBballetBbalchunaB	balancer,Bbalance.Bbakkt.BbakedBbaed,Bb3Baxe,BaweomeBawarene,Baward,Bave,B
avalanche.Bauthentication.BaureBaudaciouB
auctioneerB	attorney,BatronomicalBatonihedBatifiedBarunBartforzBarraBarmani,Barchitecture,BarcadeBarca.B	arbitrum,BappropriationBappreciatingBapple.BapologyB
apartment,BanymoreBantoBantiagoB	anecdotalBandroid.BandreiBandreen,BandeepBanarcho-capitalim,BamuementB	amplifiedBampBamountedBamoBamitBamiaB
amendment.B	ambiguityBaluminumB
altogetherBalteredBalon,BalonBallocation.BallianzB	alliance,B	alleviateBallen,Ball-time-highBalive,BalertingBalertedBalcoholBalakaBalaBaitedBairport.Baire,Bairdrop.Bai,BahtonBaheBagricultureB
aggreivelyBafety,B
aet-backedBaembleBadvance,BadrianBadobeBadminiteredB
adequatelyB	addictionBaccounting,Baccordingly,BaccomplihmentBacceptance,Bacceibility,BacBabeyBabentBabbreviationB[bitcoinB950B94B8.5B7nm,B750B7.5B687,000B669B6.6B6.5B6.4B6,500B59-pageB5000B500,B5.0B4xB4-digitB38,250B360B35.B34-year-oldB33,000B322B32,000B31tB3000B3.4B2getherB29-year-oldB26thB24-year-oldB24,000B22,000B20ac1B2026iB2026.B2024.B201dtheB201cwhoB	201cwhereB201cwellB201cuperB201cunhackableB201cuedB	201ctrongB201ctrategicB	201ctokenB201ctetB
201ctakingB
201cprettyB201cpaceB201cofficialB201cofB	201cocialB201cnationalB201cmonetaryB
201cmarketB	201cmaiveB201cmadB201clifeB
201clargetB201clackB201cinvetmentB	201cinvetB201cintitutionalB201cimportantB201chiB201chelpB201chardB	201cgoingB	201cgivenB	201cgiantB201cghotB201cgetB201cfatB201cextremelyB	201ceveryB	201ceriouB
201cenableB201cellB201cecurityB201ccutomerB201ccompletelyB201ccloeB
201cchineeB	201cchinaB201ccentralB201cbyzantineB
201cbiggetB201cbelieveB
201cbecomeB201cbaedB201camericaB	201c[the]B2019an,B2018veryB2018pamB
2018kimchiB2018freeB	2018cobraB2014theB200kB200aB2007.B2004B20+B2.7B2.0,B18.5B175B14,000B125B1200B108B103B10.9B10,500B1.7B1.0B1-5B0bc1B0645B043eB043dB041eB00edaB00a0willB00a0virtualB	00a0urveyB00a0thailandB00a0taxB	00a0tatedB00a0reportedlyB00a0omeB	00a0northB00a0networkB00a0needham:B
00a0mexicoB
00a0ledgerB00a0imfB00a0hereB00a0goldmanB00a0fourB00a0followingB00a0financialB00a0fbiB
00a0depiteB00a0cypherpunkB00a0cryptocurrencieB	00a0courtB00a0coinecureB00a0cmeB00a0centralB	00a0buyerB	00a0btc-eB00a0bitmariB	00a0augutB	00a0aboutB00a070B,B(xlm),B(xlm)B(xem)B(wbtc),B(wb)B(vpn)B(vap).B(udc).B(tb)B(ta)B(ppp)B(pow).B(pm)B(nye).B(nye),B(nydig),B(nydf)B(mm)B(mit)B(lp),B(link),B(le)B(lbc)B(jvcea)B(ip).B(initialB	(formallyB(fomo)B(finra)B	(fincen).B(fb),B(fatf),B(eip)B(dex).B(decentralizedB(dah),B(cme),B(cmc)B(cloeB(cia)B(cboe),B(cb)B(btc,B(boj)B(bi),B(avax)B(ath),B(arb)B(ada),B$8mB$81B$8.6B$8,200B$7.6B$7,300B$6,600B$6,500B$6,400B$58,354B$55B$5.9B$5.8B$5,900B$4kB$440B$4.8B$4.1B$3bB$39B$38B$36B$30,066B$3.8B$3.4B$3.2B$3.1B$280B$263B$245B$220B$210B$190B$18kB$18.5B$160B$136B$128B$12,500B$11.4B$108B#1Bzug.BzombieBzjbcBzelleBzb.comBzahnBzachBzacBzabo,ByunnanBytem:ByoungetByomiuriByieldedByield,ByiBydneyBxwBxuBxtBxpBxenaBwyreBwyoming.BwungBwu.Bwriting:BwozBwould,Bwort.Bworld:Bworking.Bworking-paceBwoodworkBwolframB
withdrawn.Bwirex,Bwinning.BwindledBwifiBwieldBwicryptBwiconinBwhitelitBwhitebitBwhintoneBwhiffBwhatapp.Bwet.B	weightingBweekly,BwebpageBwealthimpleBweakneB	weakeningBwayneBwatt,BwatefulBwarren,BwarrantyBwarned:Bwapping,BwaningBwang,BwaltonchainBwallemBwalkedBwakandaBwaitedBwahington-baedBwadeBw.Bw&kBvpcBvoyager,Bvoorhee,BvoluntaryitBvoluntaryimBvoicingBvodkaB
vocationalBvivekBvirueBvirtuBviral.B	violentlyBviitor.B	viibilityB	viewpointB	victoria,Bvice-preidentBviberBveveBverifoneBvedomotiBvc,BvayBvatlyBvarunB	vaporwareBvap.BvanuatuBvaneck,B	vancouverBvalley.BvallettaB
validator.BvacationBv2Buually,Butxo.B
utainable.ButahBurpriingly,Burge.Burfaced,BurfBupvoteBuptrendBuptoBuproarBupplier.B	upplementBupbeatBunwittinglyBunurpriingly,B	unreolvedB	unreleaedBunravelB	unnoticedB
unneceary,B
unlimited.Buniveritie,Bunique.BunimaginableBunicaBunheardBunfazedBundertanding.B
undertand,BunderpinBunderperformingBundercoringBuncontitutionalB
unchoolingB
uncertain.BunavoryBunavailableBumbrellaBuie,B	ud-peggedBubzeroBubtationBubmarineB	ubiquitouBubidedBubcribeB	ub-aharanBuarez,Bu.n.Btwo-timeB	tweettormB	tweeting:Btvl.Bturmoil,BtunningBtuitionBtudy:Btudio.Btudie.BtudBtubbornBtruth.Btrut-leB
troubleomeBtrollingBtripperBtriplingBtripe,BtrinityBtrickenBtributeBtribune,BtribeBtriangleBtrewnBtretchedBtrengthenedBtrekBtreed,Btreaure.BtreamedBtreadBtre.Btre,Btravel.Btravala.comBtrategicallyB
trannitriaBtranlation,Btranaction-baedBtrafficking,B	traditionB	tradeableB	trade-offB	townvilleBtown,Btourim.Btoronto-headquarteredBtoronto,BtormxBtormedBtorie.Btopped.Btop-performingBtonerBton.Bton,B	tomorrow,Btoll.Btold,Btoday!BtockedBtlaibBtirringBtinderBtimulu,BtimetampB
timeframe.Btime?BtierionBtier.BtidbitB	thwartingBthu,Bthroughput.BthrillerBthread.Bthought-provokingBthough:B	thouandthBthouand,BthornBthoma,Bthird,Bthing?Bthing:Bthi:BtherapyBtheorie,Bthee.B
thankfullyBthankedBthakurBtexanB	tetimony.B	terlingovBterenceBtephanBtenureBtentativelyBtenevBten-yearBteleurBteinwoldBtediouBtechnologicallyB
technicianB
technical,BteaingBteadily,BteacherBtaylorBtate-controlledB	tatartan,B	targeted.Btar,B	tanzania.Btanley.Btandard-ettingBtance,BtallinnBtalk.BtalibBtaking,Btaken,BtakeawayBtahedBtagomiBtagnantBtagedBtactic,Bta,BrvBrunning,Bruleet.BruBroyaltyBrowB	rothbard,Broom.BroofBromania.Bromania,BromaniaBrollout.BrollerBrollbackBrokomnadzor,BrokomnadzorBroenbergBrodrBrocktarBrocketedBrockefellerB	rockdale,BrocBrobberieBripio,BrinivaanBrikbank,BrightfulBride,BricheBribbitBrevokedBrevitalizedBreultantBretraceB
retorationBretetB	repurpoedB
reporting.B	reporter.B
repoitory,B
repoitorieBreplied,Brepect.BrepealBrepB
remixpointB	remarked.Brelief.BreliantB	relegatedBrektBreinventBreign.BrehapingBregulation?BrefiledBredirectBrectifyBrectificationB	recruitedBrecoupBrecord-ettingB
recipient.B	recallingBrecalledBrebuttalBrebuildBrebuffedBrebelB	rebalanceBreapingB	reaonablyBrealizationBreaffirmingBreaffirmBreader,B	read.cah.BreactivatedBreached.Bre-launchedBrawpoolBratedBrate)BraribleBrank.BramBrajBrainfallBraie.BrahidaBrage,BradaBrackedBracedBraBqueen,BquartzBquare.BqualificationBquahBq2.Bq1.Bpuzzle,BpuyBputin.BpurviewBpurveyorBpurringBpuritBpurchaerB	publicly,BptokenBpt.BpryingBproperouB
properity.Bpropaganda.BpropB	promoter,Bprohibition.Bprohibition,Bprofitability,Bprofeor,Bproecution.Bproduct:Bproduce.BprocurementBproclamationBpro-ifpB	priority.BprinterBpreparation.B	premieredBpreentation,BpreedBpree.Bpree,BpredicamentBpredeterminedBprecurorBppeBpoverty.B	potlight.Bpoting,BpotageBpot-halvingBpot-forkB	portugal,BportlandBporkBpopulaceBpopperBpool.bitcoin.comBpollingBpolicymaker.BpolicingB
polarizingBpoland.Bpoker,BpokBpoitive.BpoionBpoeingB
pocketcoinBplu500BployBplinterlandBpliegoBplexcoinBplenaryBpleadingB
plattburghBplanned,BplahB	pixelatedBpittBpitfallBpiperBpioneer:BpinningBpinnedBpin.BpilledBpillarBpiledBpierreBpie.B	pider-manBpice,Bphyically-deliveredBphrae.Bphoenix,Bphiloopher,BphilanthropicBphenomenon,B	phenomenaBphaedB
petitionedBpeteronBperonalizedB
permittingB	permiibleBperianBperceiveBpeo.Bpeo,BpenthoueBpent,B	penaltie.B	penaltie,Bpeech.B
peculator.B
peculator,BpectreB
peckhield,BpeciallyB	pecialit.B	pecialit,BpaytomatBpaytmB	payglobalBpayableBpax,BpawningBpattern,B	patronizeBpatreon,BpatohiBpatientBpath.Bpath,BpatchedBpartingBparticularly,Bparticipation,Bparticipated,Bparticipate,BparodyBparity,B	paramountB
parameter.B	paperworkBpanih,Bpanigirtzoglou,BpangolinBpan.Bpan-europeanBpan-africanBpan,B
palladium,BpainfulBpacketBpack.BoverwhelminglyBoverwhelmedBovertB
overnight,BoverhaulingBovereignty.BoverarchingBoverall.Bover-relianceBovBoutput,BoutpotB	outlawingBoutkirtBoutcryBourelveBound.Bound,Bounce.Bought-afterB	otenibly,BortizBort,B	orphanageBorigin.BorchetratingBorcaBoracle.Boptimim,Bopportunity,BopportuniticBopinedBopineBoperabilityBopera,Bopened,Bopcode.Bood,Bonward,BonnyBonewapBone-of-a-kindB	one-monthB	on-demandB	on-chain.BoliveiraBolicitedBolicitationB
olana-baedBokex.BohanaBoften,BofiB	off-limitBof]Bodd,Bocio.comBocieteBocietalBociallyBocial,BociB
occupationB	occaionalB	obligatedB
objective,B
obfucationB
obfucatingBobeyBoaktreeBoaklandBoakBoaiBnzBnye,BnuveiBnowden.Bnovoti.BnotifyB	notarizedBnordea,Bnon-tateBnon-exitent.BnipponBnihantBnietoBniche,BnicBniallBnewweekB
newletter,B	neubergerBnetwork:BnetburnB	netanyahuBnemeiBneighborhoodB
negotiatedB
neceitatedB	near-zeroBnclaBnayayerBnavyBnaviB
naturally,B	namecheapBnaira.BnadiaBnabbedB	mycelium,BmvbBmut-haveBmungerBmumbai,Bmultimillion-dollarB	multicoinBmulti-partyBmulti-million-dollarBmulti-level-marketingBmulti-facetedBmulderBmuggerB
much-hypedBmti.Bmti,BmthuliBmrBmovie.BmouthB
mountainouBmoroccoB
morningtarBmorgan,B	morgabordBmore:Bmonth:BmonteoriBmontana,B
moneygram,Bmoney:Bmoney-printingBmoneBmomentouBmom.BmoldovaBmohammedBmobtwoBmix.Bmitake,B
mirziyoyevBmiouriBmioulaBmintpalBmintableBmint,Bmiion.BmiinformationBmiileBmiguidedBmiguelBmigrantB	middlemenBmid-november,Bmid-februaryBmid-eptember,Bmid-eptemberB	mid-augutBmid-Bmicrotrategy.Bmicropayment.BmicellaneouB
miami-dadeBmfaBmetzdowd.comBmet,BmerrillBmerit,Bmerger,Bmerchandie,BmengerBmemo.BmeliaB
melbourne,BmejiaBmeenger.Bmeebit,Bmeal,B
mcdermott,B
mccoy-wardB	mcconnellBmcafee.B
maximizingBmatter:BmathematicallyB
materiallyBmaterializeB	material,BmatchedBmartijnBmarryingBmarltonB
marketpychBmarketerBmarginalBmarcu,BmarchingB	marathon,Bmar.BmaonBmanuallyBmanipulatorBmanilaBmanifetBmania,BmandelBmandate,Bmalware,Bmalmi,Bmaller,Bmale.Bmale-dominatedBmalayia.Bmaking,Bmake,Bmaintenance.Bmaintenance,Bmainnet,BmagnetBmaenaBmaeilBmae,Bma)Bm1B
luxembourgBluongoBlummi,BlucaBltcnBlozano,Blove,BlotterieBlootedB	lookalikeBlokad,BlohanBlogarithmicBloer.BlocktepBlocalcrypto,Blo,BllB
lithuania.BliteralBliten.BliquorBliquid.BliningBlingham,B	limelightBlikenedBlifelineBlieuBliechtentein.BliaBli,Blevin,BleonardoBlength.Blend,BlenBlegitBlegilature.Blegacy.BleftitBleenBledwabaBlectureB	learning,BleaningBleakingB
leaderhip,Blbank,BlazyBlawkyBlavrovBlaundryBlaunche,BlaughedBlaudBlatter.Blarimer,Blarget,Blarger,BlankaBlandlordBland.BlamentBlainB	kyungjae,BkynovaBkotaniBkorbit.BkjellBkittBkingoldBkingdom,BkimchiBkim,BkhanBkhaledBkganyagoBkeyneianBkeptic.BkenoBkeepkey.Bkeene,Bkazan,BkavalB	karagioziBkaplan,BkaneBkakao-backedBkahkariBk-iteBjuicyBjoyBjordanBjoelitBjioBjewelBjet,B
jerey-baedBjenkinBjenBjejuBjean-pierreBjaneiro,Bjail.Bjacob,Bjackpot.BixteenBix.BivanBitllBitbit,BirreponibleBironyBireland.BiowaB	involved,Binvoice.Binvoice,Binvetigation:Binvet.B
invention,BinvaionBinurer,BinultBintruiveB
intructingB	intitutedBintervention.BinterveningBinterruptionBinterpretingBinterpolBinteroperableBintermediarie,Binteraction.Binter-planetaryBinterBintenelyB	intended.Bintallment.B	intalled,BinpiringBinpireB
inolvency.Binit,Binheritance.B
inhabitantBingularB	ingle-aetBingeBingapore-headquarteredBinfringement.B	informantBinfluencingB	inflatingB
infiltrateBinexplicablyBinevitabilityBinefficiencieBindutry-wideB
indicated.B
incumbent,B	incubatorB	incluion,B	included,BinboundB
inacceibleB	in-demandBimultaneouly.BimulatorBimtokenBimpreBimplyingBimpliticBimpleledger.infoBimperialBimpedeBimoB	imminent.Bilver.Bii,BignoredBigniteBignificant,Bideal.B
ide-by-ideBicapBiblingBiberianBiacoinBhydropower.BhycmBhutleBhunt.Bhungary,B	hungarianBhumeBhughBhtmlBhruggedBhqBhourlyBhotlyBhortenedB
hortcomingB
hort-livedBhore,BhopifyBhookedB
honeypointBhondaBhonanBhomageBholytranactionBhogegBhoegnerBhodl,BhockeyBhkreliBhkdBhitterBhinyBhinedB	hindight,BhinderBhihidoB
highlight.BhiatuBhenderonBhelloBheld.BheightenBheight.Bheight,B	hegemony.BheathBheaterBheart,Bhealthcare,BheadacheBhazardBhayeBhawaiianBhatilyB	hathaway,BhatenBharvetBhartBharoldBharneBharma,BhardingBhardetBharamentBhappierBhakepayBhahmakBhadowyB
hackathon.Bh-e-bBguo,BgunningBgullibleBguild,BguidedBguide.Bgud,B	guardtimeBground.BgrindBgref,Bgreater.Bgreat.Bgreat,BgrabbedBgpu.Bgovernment-controlledBgone,Bgolix,BgodzillaBglobalizationBgleanedBglaringBglance.Bgiving.Bgiven.BgitcahBgiletB
georgetownBgenreBgemtoneBgemBgdprBgdBgcahBgbtc.Bgbtc)Bgbp.B	garneringBgarlinghoue.BgalaB	furnitureB	funnelingB	fundtrat,Bfundamental,Bfunctionalitie.Bfully-licenedBfueBftlBfrutration.BfrothyBfrotBfrontmanBfront,BfrohmanBfrequently.BfrenoBfreetonB	fraudter.BfrankoBfrankfurt-baedB	francico.Bfrancico-headquarteredBfragileBfraerB	four-yearBfoundry,BfortniteB
fortnight,BfortB
formation.Bforklog,B
forgettingBforget.BforeightBforecat,BforageBflyerBfluctuation,BfloweeBflikBflieBfleeingB	fledglingBflaggingBfixtureBfixatedBfive-dayBfintech.Bfinland,Bfinder,BfihingB	fictionalBfiat-to-cryptocurrencyBfervorBfertileB	ferruccioBfendB	federatedBfearedBfdicBfavor.Bfault.Bfather.BfatalB	fat-pacedBfamiliarityB	falifyingBfalifiedBfale.Bfale,BfaireyBfair.BfadBfactorieBface-to-faceBface,Bf2pool,Bf2poolBey,BextraordinarilyB
extortion.BextortB	extenibleBexploratoryBexploration,Bexplaining,BexpiryB	expenive,BexpendedBexpectation,Bexpect.BexpaneB
exorbitantBexodu,BexmaBexited.Bexhibition,B
execution,BexcluionBexceptionalB
exception,B	evolving.B
evolution.BevmB	evidence.Beverywhere.BeveredBever-expandingBevaluation,Beurope-baedBeuphoriaBeunBeu.Bettled.Betp.Betp,Betonia-baedB	ethereum?BethercanBetc.,Betback,BetbackBervice:BerdoganBequivalent.Benvironmental,BenvionB	enviionedBentiretyBentail.BenlightenedBengine.BendureB
encryptingBencodeBenamoredBenactingBeminar,Bemiconductor.B	emefiele,B	embeddingB
embarraingB	embarkingBelviraBelviBelf-profeedBelf-managedBelf-madeBelf-decribedBelevatedBelected.Belaborated.Belaborated,B	el-erian,BeinteinBeimanBeighteenBegypt.BeggBegBefficientlyB
effective.Beem,BedelmanBecurely.Becretary-generalBecrecy,BecophereBeconomizingBecond-highetBeco-friendlyBeclipedBechoedBechoB	eccentricBeattleBearn,B	earmarkedBearBeaier,Be-kronaBe-giftBe-cediBdwindledBdutinBdunamuBdumbBdue.Bdrink,BdrillingBdriftBdraper,BdraghiB	downrightB	download.BdougBdoubt.Bdotcom,Bdoor.BdoomedBdonBdog.BdodgeBdo?BdmccBdivineB	divertingBdiverification.BditrutB
ditinctiveB	directly,BdiplomatB	diplayingBdiparityBdiolveB	diminihedBdiminihBdimiiveB	dimantledB
diligentlyBdihBdigorgementBdigitalizedB
digitalizeBdifficultie.B
diego-baedBdied.Bdie.BdidiBdictatorhipBdicriminationB	dicoverieB	dicountedBdicontinuingB	dicerningBdibureBdiappearingBdiappeared,BdiagreedBdiablingBdewurteBdevelopmentalB
developed.B	devaluingBdevalueBdevaluation,BdethroneBdetermined,B
detabilizeBdepreciatedB	deployed.B	deployed,B	depictionB	dependantBdenominatorBdenityBdenied.BdelvingBdelugeBdelphiB	deloitte,BdeloitteBdelo,Bdeliberation,BdeletingBdeired.Bdeigner,Bdefipule.comBdefeniveBdefene,BdeedB	deductionBdecipherBdecimalBdeceaed,BdecB	debt-baedBdebaingBdebaeBdeaBdavi,BdaviBdaqingBdao,BdankB	dangerou.BdanaBdampBdamagingBdaa.Bda,Bd10eBcyclebitBcurtailBcurryingBcurateBculminatingBculianuBcuban.BcryptoqueenB
cryptoniteB	cryptomktB	cryptojobBcryptograffitiBcryptocurrency-infuedBcryptocurrencie;Bcryptoconomy,Bcrypto-tradingBcrypto-themedBcrypto-priceBcrypto-currencyBcrypto-currencieBcrypto-centricBcryingBcrutiny,BcrumbleBcrowdourcedB	crowdale,Bcrowd,BcrollB	criticim.B	criteria.Bcredential.B	creative,BcrawlBcraveBcratchBcrappedBcrapBcrambleBcrahe,BcrabbleBcpbBcozomoBcourier.Bcounter-terrorimBcottonBcotiaBcorrectionalBcorrect.B
corporatitB
corporate,BcorazonBcopper,BcopperBcopingB	copernicuBcooterBcoordinator,BcoordinatorBcoordinatingBcooperation,Bcool,B	conveyed,BconvenientlyBconvenience.B
conulting,B
conultant,Bcontroverial,Bcontributor,B
contratingBcontraryB
contrarianBcontentiou,BcontellationB
contactingB
conortium,Bconolidation.Bconolidation,Bconnection,BconnecticutBcongrewomanB	confuion,BconfueBconformBconfeedBconeny,BconecutivelyBconduitBconcurrently,B
conceptualBconceionB	concedingB
concealingB
compromie.BcomprehenivelyB
comprehendBcomplicated,B
complicateB
compliant,Bcomplex.BcompileBcommuterB
community:Bcommunity-ownedBcommunicatingB
commented,BcomedianB	combined,BcombedBcombBcolumbuB	columbia.Bcollege,B
collector,BcointrackingBcoinplugBcoinnetBcoinmeBcoinifyBcoindekB
coincidingBcoincidenceB
coincheck.Bcoin.dance,B	cofoundedB
cofound.itBcocaine,B	coca-colaBcobra,Bco2B
co-managedBco-hotBclovrBcloure.Bcloud.Bcloud-miningB	clothing,B	cloed-endBclinicalBclimate,Bclearinghoue,B
clarified:BclanBclampingBclampBclaic.BcivilizationB	circulateB	circular,B	cinematicBcience.B	chweikertBchwabBchritma.BchipotleBchiff.Bchicago-baedBchewBcheriBchedule.Bcheaper,Bcheap,Bchaum,BchatexBchat.Bcharter,B
charimaticBcharacterizeBcharacteritic.B	changetipBchanB	champion,BchairedBcex.ioBcentricBcenorhip-reitant,BcbaBcavengerB
cautioningBcaucauBcauallyBcattleBcatteredBcatingB	catherineB	category.B
categorie:BcatatrophicB
catapultedBcatalanBcat,Bcarter,Bcarten,Bcartel.B	carpenterBcardano.Bcaraca,BcanvaBcanineBcandal.BcancerB	canceled.B	canceled,BcanaryBcampbellBcameronBcambotBcamachoBcalpBcalable,Bcalability.Bcaino,BcaillB	cahfuion.BcahaddrBcah?Bcah:Bcah-outBcah-inB
cah-focuedBcah-acceptingBcafBcae:BcabBc+Bc#Bbuying.Bbut.BburnikeBburned,Bburn.BbuoyedBbumpedBbumpBbullion,BbulletBbull.BbuhariBbuhBbucketBbtc/bchBbrutalBbruhedBbroker-dealer.B	broadcat.BbritonBbrightetBbrewing.Bbrewdog,Bbreak.Bbreak,BbrazenBboworthBbowBbottom,Bboth,BborneB	borderle,Bboom.BbookedBbook:Bbonue.Bbogart,BbmgBbloodyB	blockwareBblockfi.BblockeerBblockchain-centricBblatBbityardBbityB	bitrefillBbitmaxBbitinkaB
bitfinexedBbitcoin!BbitcoiinB	birthday.BbirdBbirchBbip39BbingBbig-nameBbiboxBbiaedBbi,BbernteinB
berner-leeBbermanB	berkeley,BbenoitBbennettB	bengaluruBbeneficiarieB	bemoaningBbemoanBbeltrBbelt,B	believed.Bbelieve.Bbeing.B
behavioralBbehavingBbehalf.Bbegun,Bbeen.BbchgBbch/ud,Bbch/eurBbay.Bbattle,BbatteredBbatcheBbarronBbankruptcy.Bbankruptcy,B	banknote.B	bandwidthB
bandwagon,Bbancor,BbancaBball,BbalkanBbalboaBbahBbael,Bbad.B	backdatedBbacelarBbabieBayondoBaxaBawemanyBavertBavedroidBautria.Bautria,B	autonomy.BautonomyB
automotiveBauthenticatedBauthenticateBaumB	attached.BatopBatomBatohipayBatohilabBatifyingBathlete,Bat,Bat&t,BartiticBarreted,Barrangement.BaroeBarmeniaBarie.BarguBardentBarchivedBarca,BarbitrarilyBarbicorpBarbBaquagoatB
approache.Bappreciation.B
applicant,B	applianceB	applaudedBappeaeBapparently,B	apparatu,BapolloBapmexB	apirationBapect.Banymore,Banwer,Bantonopoulo,B	anti-viruBanti-blockchainB	anonymou,BanonBangel,B	anchorageBanarchy.Banarcho-capitalitBanalyt:BanalogouBanadoluBamerBamberBamanBam.BalteringBaltairB	alphabay.BallureB
allocatingB	alliance.Ballege.BalienB	algorand,BalfredBalbum,BalbertoBakerBak,BajayBairingBahleyBahimBagutBaggregatingBagenda,Bage-oldBagain?B
afterward.BaftBafraidBafloatB	affirmingBaffiliationB	afedollarBadvertiement.BadverelyBadult,Baddreed.Badd?Bada,Bactive.Bactivation.B
activatingBacreBachievement,BacdBaccordingly.BaccommodatingBaccidentB	accepted,Babra,Babout?BaborbedB	abeychainBabc.Baave,Baange,B	aainationBa.m.,B[inB[ic]B[cryptocurrency]B9th,B9.99B8:30B8-kB750,000B70,470B7.7B7.4B7,500+B7,000+B630,000B620B60.B5.8B5.6B5.5B5.2B5-10B5,700+B48-hourB40.B4.9B4.7B4.6B4.2B4,000+B393B38.B38-year-oldB35,B336B32mb,B30th.B3.7B3.5nmB3)B295B270B260,000+B256-bitB24hB228B20th,B2023,B201dbitcoinB201d:B201d),B201czeroB	201cwhichB201cwhetherB201cwallB201cupectedB201cunprecedentedB
201cunlikeB
201cunitedB	201cunionB201cuerB201ctwoB201ctranactionB201ctockB	201cthreeB201ctechnicalB201ctayB201cruiaB	201crightB201cregulationB201cprotectB	201cpriceB201cpayB	201courceB	201corderB201coil-backedB
201cnumberB	201cnorthB	201cneverB201cmr.B201cmoveB201cmainB
201clivingB201cleadingB201cleB201clawB201cinvetingB201cinflationB201cindutryB201cindividualB201cimproveB	201cimpleB201cignificantlyB201chundredB201charkB	201cgreenB201cfurtherB201cfullB201cfourB201cfounderB	201cfinalB201cfamiliarB201cextremeB201ceverybodyB201cetB
201cenergyB201cendB201cemergingB201cemergencyB201ceekB201cecuritieB	201cecondB
201cdoubleB201cdeignedB201cdefiB	201ccyberB201ccrypto-aetB201ccriticalB201ccriminalB201ccreatingB	201ccraigB201ccompreheniveB201ccomeB201ccamB
201cbeyondB201cbeB201cbchB201cbanningB
201cattackB201canonymouB
201calonzoB	201calmotB201cadditionalB201caccordingB
2018zombieB2018whatB2018topB2018tablecoinB2018realB
2018nobodyB	2018geneiB
2018cahingB2018buyB2001.B2000,B2-10B2,700B2,609B2,300B1mB1:00B1999,B1996B1992.B1990B1980B1973B1970,B191B19,000B187B184B165B161B16-yearB14.3B13th,B135B128B127B123B112B111,000B11-yearB10kB101B100kB0bbfB0b95B0633B0631B0629B0433B0153urB
00fcndchenB00eateB00e9tB00e9raleB00e9monB00e0B00a0yourB00a0whatB	00a0underB00a0theyB00a0thaiB00a0telegramB00a0technologyB00a0robinhoodB00a0recentlyB	00a0polihB00a0petitionB00a0p2pB00a0onlyB
00a0onlineB00a0oneB00a0netherlandB00a0naimB	00a0nadaqB00a0mt.B	00a0moneyB
00a0miningB
00a0luxuryB00a0localbitcoinB00a0leadingB00a0launchedB00a0iranB00a0invetmentB00a0inceB00a0however,B00a0hongB00a0homelandB00a0hapehiftB00a0gettingB00a0fivebuck.com:B00a0firtB00a0fidelityB00a0developerB00a0currencyB00a0coingeekB00a0citibankB00a0canB
00a0calledB00a0butB00a0btccB00a0bittampB
00a0becaueB00a0beB00a0backB00a08B0.21.0B0.17.0B(zec)B(xlm).B(wj)B(vr)B(vat)B(utxo)B	(unacrip)B(uchB(tx:B(tx).B(tranactionB(roi)B(rbf)B(pv)B(ptr).B(pmcB(partB(oecd)B(notB	(nationalB(mpc)B(mp)B(mou)B(je)B(j/th).B(im)B(ifwg)B(iamai)B(ia)B(hw)B(hortB(gbtc).B(gbtc),B(fiu),B(fc).B(fbi),B(eu).B(etn)B(ethB(eo),B(ema)B(egwit).B(eeB(ebi)B(dyp)B(dny)B(dnm).B(dlt).B(dh)B(dea)B(dapp).B(daa)B(d)B(cz),B(cz)B(cpu)B(cer)B(butB(btg),B(btc1)B(bok)B(bog)B(boa)B(bafin)B(aroundB(arkB(approx.B(amm)B(aa)B(2fa)B(20),B$95B$9,700B$87B$86B$815B$80,000B$8.1B$72B$70,000B$7.7B$7,400B$68B$640B$6.5B$6.3B$6.2B$550B$500k.B$5.5B$5.4B$5.2B$4mB$41B	$400,000.B$4,000.B$32,000B$310B$30mB$30k.B$3000B$3.6B$3,870B$2kB$28,600B$260B$26B$257B$243B$235B$20mB$200kB$20,000,B$2.45B$1kB$1bB$195B$185B$180B$18,000B$16,500B$14kB$135B$13,500B$12.5B$12.4B$110B$103B$10.5B$1,800+B$1,800B$1,300B$Bztorc,BzoomBzooBzodiaBzimwaraBzimbabwe-baedBzielkeBzhuoer,Bzero-feeBzermattBzenBzeger,BzdnetBzclBzambianBzahabiBzabuBzabo:ByveByunnan,Byria,ByriaByovopayByour,ByougovByouefByorkerByork-headquarteredByoonByong-jinBynthetixBynchronizationBynbioticBymbol,Bygnia,Byet?ByemiB	yeewalletByeeByearningByear-to-date.ByardBxwapBxunleiBxtzBxpayB	xinjiang,BxboxBx3Bwyoming,BwydenBwuhan,Bwu:Bwrongdoing.Bwrongdoing,Bwritten,Bwozniak,BworriomeBworrie.B	workplaceBworking,B	workflow.Bworker,BworeningBworenedBwoolardB	woodwork,Bwood.Bwonder,Bwomen,BwjBwithtandBwithholdBwiringBwiquote,BwindleBwindexBwin,BwimeijerBwilhireB
wikipedia,BwiinfoBwift,B
widomtree,Bwide.BwicoinBwi-madeB	wholealerB
whitelitedB
whatoever.B	whaleclubBwex,B	wetminterBwere.Bwell-regardedBwelfareBweiboBweepBwedeBweddingBwebpage,BweaveBweakenBwbtc.BwazilandBwater-cooledBwatche,BwatcheBwatch.Bwatch,BwanedBwamyBwampedBwaltBwalmart,Bwallet)BwalhBwale.Bwale,Bwait,BwagnerBwageredBwagedBwBvulnerable.BvowBvoice.BvoellBvitaeBviru,B
virtually.BvillainBvijayBviit,BviionaryBviennaBvideo:B
vice-vera.BviacoinBvgcBveteran,B
veratilityBveracityBver.B	venturingBventuredBventupBvenrock,Bvenmo,Bvenezuelan,BvenBveilBvega.BveeBvaulttelBvault12Bvat.BvarieBvariant.BvantageBvanihedBvalveBvalrB
validator,BuzhouButBurvivorBurrogateBurreptitioulyBurged.BurfingB	uquehannaBuptimeBuptateBuptakeBuprnovaBupriingB	upremacy.B	upremacy,Bupon.B	uploadingBupide.Bupiciou.Buphold,B	upgraded,Bupgrade:BupfrontB
uperviion,Bup:B
up-to-dateBup-and-comingB	unwittingB	unwillingB	unuccefulB	untouchedBuntoldBuntappedBunrulyB
unreportedB
unrealiticB
unpublihedBunofficially,BunnyBunloadBunleahBunknown,BunitingBuniquelyBunionpayBuninterruptedBunilBuni,B	unfolded.BunfairlyBunettingB
unemployedBuneenBundinB
undeterredB
underwaterB
underneathBunderground.BundercutBunder.B
uncrupulouB
uncoveringBuncollateralizedBunclear.Bunclear,B	unclaimedB	unchartedB
unchanged,BuncertaintieB
uncertain,BunbelievableB	unbanked,BunavoidableB	unanweredBunanctionedBunacripBum.Bum,BukhavatiBuiBuhiwap.BuheredBugarBuganda.BuezBuefaB	uccumbingBuccumbedBubi,B	ubcriber,BubcribedBuanBuae,Bu?Bu.kBtyonBtyledBtyingBtwo-partB	two-monthBtwitter:Btwit.Btwit,Btwin,Btwice.Btwice,Bturn.BturboBtuniianBtungBtumbler,Btumbled.Btudio,Btudent.Btucker,BtuartBtrugliaBtrueud,Btrouble.BtroopBtrongu,Btrip.BtrioBtrike,BtrickedBtribalimBtrialingBtrialedBtrength,BtrefiBtrebleB	treaurer,B
treatment.BtreamliningBtrc20BtraveledBtratophericBtranpoeBtranparent.Btranmitter.BtranmitB
tranition.Btrangreion.B
tranformerBtranferred,BtranferableBtrancheBtranaction;BtramB	training,Btrain,Btrail,Btrafficking.Btraditional,Btrack,BtoyotaBtournament.Btour.B	touchdownBtotaledBtortenBtortataB
torrentingBtornBtorch.BtoppledBtopologicalBtop-tierBtoolboxBtone.Btoll,BtoiletBtohiBtobaBtirelelyBtir.BtipulateBtippr,BtipprBtip-cardB	tinkeringB
timulatingBtimetampingBtimedBtimecoinprotocolBtime-lockedBtime).Btill.BtiglitzBtie,BticeBtiBthree-quarterBthoroughbredBthomponBthird-generationBthieve,Bthi?BtheymoBthetreet.com.BthermalBthereof.BtheoremBthankingBtetragonBtetnet,BteterB	terrorit,BterribleB	terraformBterabBtenxBtenuouBtenniB	telanganaBtefanonBteeringBteepedBtechxBtechnology-baedBtechnoB
technique,Btear.BtealerBtea,Btbilii,BtbdBtaveBtatiticallyBtatitBtating,BtateleB
tate-iued,Btate-fundedBtataBtarvedBtarredBtaroB	targeted,BtareckB	tanzania,BtameBtallionBtalled.Btalent,Btaleb,BtalaBtakeholder.Btakeholder,BtaiwoBtaintedBtahkentBtah,Btag.BtaeBtadiumBtablet.BtabacBt-hirt.BryukBrydeBrxBrwanda,BrvnBrut,BrundownBrumourBruin.B
ruian-bornB	ruia-baedBrugBrueBrudeBrout.B	round-up.B	rotterdamBroom,Broof.Broll.Brole-playingB	rogozinkiBroengrenBrockdaleBrock,BrobutneB	roboforexBrivneBrival.BriteB	ripplenetBriot)BringgitBring.BrikierBrik-onBrik-freeBrik-adjutedBrightlyBright:BrifleBrieder,BriederBriddellBrichmondBrhodeBrexBrevolvedBrevixBreviitBreviingBrevampB
reurrectedBreurgentB	reumptionBreubenB	returned,B
retropect,B
retrievingBretored,Bretirement.BretireeBretingB	retainingBret,BreputationalBreputation,BrepurpoeBreponiveB
replicatedBrepectivelyB
repectableBreort.BrentingBrenterBrental,B	renovatedB
renouncingB
renewable,Brenbtc,BrenamedBremunerationBremoved,B
reminicentB	remarked:B	relocatedB	relocate.BrelinquihingBreligionBrelieveB	reliable,B	relevant.Brelentlely.BrelaxedB	relative.B	rejected,Breitant.B	reintatedBreintateB
reinforcedB	reinforceB	reilient,BreidencyBregulatory,B	regulate,B
regitrant.B
regitered.B	regardle,Brefugee.B	refrehingBrefrehedB	refinanceB	refinableB	reemblingB	redeemed.BredactedBrecycledB
recreatingB	reconveneB	reconcileBrecommended.Brecommendation,B
recheduledB
receiving,BrebootBrebateBreapedBrealtor.comB
real-time?Breading,B	read.cah,Bre-evaluatingBre-enablingBraymondBravageBraulBratio.BratBrarible,BrariBrapidly,Branked.Brampant.BralphBrajanBraizBraionBrainyBraie,B	radicallyBracibBrachelBracerBrabobankBr/walltreetbetB	r/darknetBr/btc.Br/btc,BquorumBquonticBquo.BquizBquinnBqueue.Bqueue)B
quarrelingBquantifyB	quantifieBqualifieBquahingBqiwi,BqiwiBqin,BqianBqe,Bq4,Bq3.Bq1,BqB
pychology.BpuzzlingBpuzzle.BpurpoelyBpurgingBpurevpnBpure.BpunihedBpuneBpuhbackBpudB
publicity,BpteriaBprudentB	proximityBproxieBprovokeB	proviion.BprovableB	protocol)BprotetedBproppingBpropoition.Bpropoition,BpronomoBpromoBprominentlyB	promiing.Bproject:BprohibitiveBprohibited.BprogrammaticB
profoundlyBprofit-takingB	profilingB
proecutingB	proceing.Bproblematic.Bpro-democracyBpro-bitcoin.BprizedBprize-winningBprize,B
privateflyBprivacy:B
prioritizeB	printing,Bprint,BprincipallyB	princetonBprimerBprime,Bprim,BprimBpricing,BpricewaterhouecooperBpreyingB
previouly.B
prevalent,BprepperBprepB	prejudiceB	preented,Bpreence.Bpreence,B
predicted.B
precipitouBprecie.Bpre-regiteredBpre-halvingBpraoBprague.Bpraad,BpraBpoyntB	powerful.Bpotter,B	potponed,BpotionBpot:B
pot-mortemBpot-effectiveBportedBpornographyBporadicBponzi.Bponzi,B	pontaneouBponderedBpoloB
political,B	polinelliBpoleB
pokeperon.BpokaneB	poitivityBpoioningBpoibilitie.Bpoeion.Bpoe.B
pocketnet,BpncBpmcB	plutoken,Bplummet.Bplugin.Bplugin,BplottingBpleaureBpleaantBplaylitBplayableB	planetaryBpivotedBpitolBpiralingBpiquedBpionexBpinkBpingBpin-offBpilot.Bpiked,BpieterB
pider-man,B
philoophy.BphilippeBphilanthropyBphiladelphiaBphihing,BphenomenallyBpharmaceuticalBph.d.Bpetro-dollarB
petitionerB	peterffy,B
peterburg,BpetahaheBpervaiveBperuvianBperuadedBperuadeB
perpetuateB
permeatingBperkB
peritentlyBperitedBperiodicallyBperhap,BpenionerBpenceBpeinBpeimiticBpeg,BpeeledBpeekBpeedyBpeecheBpedroBpeddleBpeculiarB	peckhieldBpecification.BpecialtyBpboc,Bpayout.Bpayout,BpayfatBpaycoinBpatriaBpatent-pendingBpatelBparrieBpark-headquarteredBpariqBparent.BparedBparanoidBparalelnBparaiteB	paradigm,BparadeB	parabolicBpandemic-relatedBpanama.Bpanama,BpaltrowBpakitan.Bpaion.B	pain-baedBpaid,Bpagni,BpagBpacaoBpacBpaageBozBowBovervalued.BoverubcribedB	overtock,BovertlyB
overtatingB	overtatedB	overruledBoverreachingB	overreachB	overratedBoverloadBoverlayB	overhypedBoverheadBoverdueBoutragedBoutlook:BoutlierBoutizedB	outhiningB
outheaternBoutage,Bout:BourcingBour,BoundedBoundbiteBounce,BoudBorry,Boro:BorleanBorganiedBoregonBorderlyBorbit.Bor,Boptimization.B
optimitic,BoppreionB	opponent.Boppoite.Bopined.Bopined,B	operativeBoperational.BopentimetampBopening,Bopen-ource,Bopen-mindedBonward.BontierBontario,BonoBong,BoneelfBone-topBonce,Bon?Bon:BomeingBolympic.Bolved.BolidifyBolicitBoldex.aiBolana.Bol,BohoBoften.Boffene,B	offender.BodeyBodd.BocioeconomicBocio-economicBocializeB	ocialite,Boccurrence.B	occurred,Boccur,Boccaionally,Bobviou,Bobtacle,BoboleteB
objection.Boberved,BobeedBoar,B	nye-litedBnydB	nurturingBnurturedBnuringBnuclear-poweredBnowhere,Bnow?BnoviceBnovel.BnotionalBnothing.Bnothing,BnotarizationB
notabilityBnorm,BnonghyupBnonene,B
non-miningBnon-mineableBnon-governmentBnon-diviibleBnon-cutodial,B	nominatedBnomiBnoeBnode.j.BnodBno-dealBnmBnimeraBnimbuBnikolaiBnikitaBnihithB
nightmare,B	nigerian.BnicknameBnickelBniche.BnextechBnexo,BneweggBnewdek,BnewbBnew.bitcoin.com:BneviB	neutrino.B	neutrino,Bnetwork)Bnetherland-baedBnetcentBnervouBneo.B	neighbor,B	near-termBncube,Bnba,B
navigatingBnaverBnatyBnatwet,Bnationwide.BnarrowlyB	narrowingB	narcotic.BnamibianB	nakamoto:BnakaendoBnailBnagBnabiullina,BnabBmytery,BmycryptoBmutconBmurderedBmunger,Bmultiplier,BmultilateralBmulti-cryptocurrencyBmuicalBmuic.BmuggledBmug,BmugBmucleBmuch-vauntedBmuBmtvBmozillaBmow,Bmount.BmouB	motorportB
mot-tradedBmot,BmorphingBmoroB
morningideBmoonpayB
monumentalBmonopolizedBmonitoring,B
money20/20BmomintB	mom-n-popBmoheB	modernizeB	moderatedBmode,BmodB
mocow-baedBmockingBmockedBmobilityBmobile.Bmnuchin,BmmcBmixer,BmiundertandingBmit,BmirokiiBmiracleBminereumBmindedBmillion?Bmilitary-indutrialBmildBmilaBmihutinBmifidBmidummerBmidnightBmid-octoberB	mid-june,B	mid-july,Bmid-january,Bmicrotranaction.B	microoft.BmicronationB	microchipBmicro-tranactionBmicro-invetingB
miconduct.BmickyBmgtiBmgmBmezzoBmeyerBmewBmerit.BmergingBmercuryBmerculetBmercierBmentorB
mentioned,Bmention,BmendBmempool.Bmempool,B	meme-coinBmelkerBmeldBmei,Bmegaupload.B	medvedev,Bmean?BmckenzieBmcalileyBmcBmb.Bmb,Bmayor,Bmaxwell.BmaurerBmaturingBmatrixport,Bmaterialized.Bmater.BmarketedB
marketableBmarket-leadingBmarkerBmarieBmarginedBmappingBmaoBmanufacture,BmanpowerBmanningBmankindBmanipulativeBmanifetoB	mall-caleB
malevolentBmaking.BmakeupBmakerplace.Bmajority-ownedBmajor,BmajeticBmaively,B	maintain.BmaiieBmaieBmahingBmahhadi,BmaherB	magicallyBmaggieBmaerkiBmaccoin,Bma,Bm1911Bm,BlyuBlylianBluxury.Bluxembourg,BlurkingBluno.BlunaBlumped,Blump.BluminaryB
lucrative.Blubin,BlubinBlower,Blowe,BlovingBlopp,BlonBlogic,BloggingBlodeBlockupB	locked-inBlocated.Blocally.BlobbyitBlobbyBload.BlmaxBlivreBliving,BlivecoinBlive-treamingBliu,BlitenedBliquidation,BlippingBliongateBlionelBlioncomputerBlindayBlindaBliliaBlikenB	lightpeedBlighthoue.cahB	lifetyle.Blide,BlicenceBlibrary.BlibrarieB	libertad,BlibertadB
liability.BlevyingBlelantuB
leibowitz,BlegitimatelyBlegitimate.Blegitimate,Blegitimacy.BleggBlegend,Blegal.Bleft.B
leer-knownBlee.Blebanon,Bleave,BleanBleaked.B
leaderhip.Bleaderboard.Ble,BlawfulBlaunche.Blatam.Blat?Blat,B
languihingB
landlockedBlamtek,Blamborghini,BlambertBladderBlacroixB
lackluter,BlabelledBlabelingBl2BkyrtenB	kyrocket.BkyberdmmBkwh.Bkweli,Bkutcher,BkunangBkumar,Bkubitx,BkuamaBkualaBkrill,BkrebBkrawiz,BkranjB
korea-baedB	knowinglyBknottBklineBkleinBklauBkit.Bkit,Bkinea,BkilogramBkiloB	keywalletBkeytoreBkeier,BkazakhBkayamoriBkatingB	katherineBkateboarderBkarpBkarolBkarate-chopBkaraokeBkandiabankenBkahkari,Bjut,BjungBjuliaBjukebox.cahBjukeboxBjrBjpy,BjpegBjourney,B
journalit.BjorgeBjone.B
joinmarketBjobleBjewelerBjeop,BjeongB	jefferie,BjefferieBjean-michelBjauneBjargonBjarBjanataBjafariBjaB	ix-figureBitcomBitanbul,Bironically,Birkutk,BirkutkBir-ciBiphoningBiphone.Bip.BiorioB
invocationBinvite-onlyBinveted.B
invention.B	inventingB
invariablyBinvadingBinvadeB	inundatedBintructBintroductoryBintroduction,BintroBintrepidBintra-rangeB	intra-dayB
intitutingB
interview:B
intervenedBinterplanetaryBinternecineBinterminableBinteret-payingBinterconnectedBinter-governmentalB
intention.BintelligentBintall,Bintact.Bintability.Bint.Binput,B
inpection.B	inpected.BinpectedB
innovatingB	innocenceBinkingBinkedB
injection.B	injectingB
initiationBinited.Binight.Binider.B	inheritedBinheritB
ingredientBinglingBing,B
informaticB
informallyB
influence,Binflow,B	infinity:BinfiltratingBinfiltratedB	infectingBinfancy.Binevitable,BinertBinefficiencyBindutrialitBinducedBindriBindiputableBindictment,B	indicate.Bindia:Bindexe,B
incubationB
increaing,B
incorrect.BincoreB
including:B	included.BinclinedBinciteBinchingBincarnationB	incapableBin-tore.B	improviedB	improved.Bimprionment.BimprionmentB	impreive,BimpracticalB	importingBimportationBimportance,BimploredBimplicitBimplication.BimplerB	imperviouBimperonatorBimperonationB	impatientB	impacted.BimoneBimmoral.BimmoralBimmon,Bimagination.B
imageboardB
im-wappingBilyaBiluanovBillyBilluminatedBillinoi,BilgizBilbert,BignumBignored.Bignificant.B	ignallingBign-upBign,BiftBieo,BidoBidiotBidex,B	ideology.Bidentification.Bidentification,BidentifiableBidehift.ai,B
idehift.aiBiddharthBid,Bicorating.comBichuan,Biceland.BicebergBibrahimBibm.Bibling,Biberia.BibenaBhydro-quebecBhuvalovBhuttleBhunningBhumorouBhumanitarianBhuang,BhuangBhu,Bhtc,BhrfBhrewdBhrem,BhredBhrapnelBhrBhoulderBhotpot,BhotilityBhoted,Bhort-lived,B	horowitz.Bhore.Bhope,BhonkB	honetnodeBhonet,BhonedBhoneBhometeadingBhomeroB	homegrownB	homebuyerBhoffmannBhodl.Bhive.oneBhitch,Bhitbtc.BhiroakiBhinmanB	hinderingBhinderedBhin,Bhike,Bhigh?Bhift.BhibawapBhiatu,BheroBhere?B	heraldingBher.B
hemiphere,B	hemiphereBhelter-in-placeBhelterBhelinki,Bheld,B	heitationBheit,BheepBheedBhedge.Bheat.Bheard.BhealingBhealBheadyBheading.BhdfcBhatedBharuhikoBharply.Bharply,BharneedBharmonizationBharing.Bharing,B	hargreaveBharepotBhareholder,BhardyBharding,B	hardeningB	hardcodedBharaedBhappineB	hapehift.B	hangzhou,Bhanghai-baedBhanenBhandwrittenBhamperedBhammahBhalving?Bhalt.B
halloween.BhallowedBhakirov,BhakeupBhah,Bhadow,Bhad,Bh1Bh-antBguidoB	guardian,BguardedBguanBguaidoBgrowth:Bgrown.BgrolyBgrin.Bgrin,BgrilledBgriggB	grievanceBgreypBgreetB
greenwood.B	greenwoodBgreenlightedBgreed.Bgreece.Bgreatly.BgrapingB	graphicalBgranted,Bgrant.BgrBgpkBgovtBgorgeouBgopelBgoogle-ownedBgoodwillB
gooddollarBgoing,BgogoBgoe.BgobbledBgmt.Bgmo,BglutBgloomBglitch.BglancingBglance,BgladteinBglacierBgiutraBgithub,B	gigablockBgigBgift,Bghot.Bget,BgermanovichBgeographicallyB	genuinelyBgenevaB	generoityB
generator.BgeneraliBgear,Bgdp.Bgdp,BgdlcfBgdlcBgbBgazprombankB
gatekeeperBgatecoinB
garavagliaBgarage.BgaparinoBgap,BgangnamBgang,Bgametop.Bgame-changerBgalileoBgaidarBgacktBg20,BfvniB
furtheringBfuriouBfunnelBfunkyBfunkoBfungibility.Bfundraiing,Bfull-lengthB
full-blownBfuenteBfuel.BfrutrateBfrozen,B	frontier,BfriedBfrickBfreightBfreeze:Bfreeze.Bfreero.org.Bfree:BfraxBfraughtB
frankfurt,BfrancecoBfractionalizedBfpgaB	four-hourBfound:Bfortune:Bforfeiture.B	forfeitedBforex,B
forerunnerBford,B
forbiddingBfootageBfooledBfood.BfondneBfomo3d,Bfomo,Bfocu.BflowedBflouriheBflourih.Bflourih,B	florence,BflorenceB	floodgateBflirtB
fleetinglyBflavorBflaringBflameBfizzledB
five-digitBfiroBfiringBfireworkBfirewallBfirefoxBfinney.Bfinney,BfinervBfindbitcoin.cahB
financing:BfilthyB	filteringBfilteredBfilmingBfilmedBfiled.Bfiled,BfilebaeBfiguringB
figureheadBfiguredBfight,B	fiendihlyBfiction,Bfiaco,BfiaB	feverihlyBfeverihBfeveredBfeudB	fetivitieB	fernandezBfemale.BfelonyBfelonieBfeed.Bfeature-richBfeature-lengthBfearfulBfca,B	favorite.BfauxBfaucet,Bfater.Bfate.Bfat-growingBfaroBfarmer,BfariaBfaredBfantomBfantaieBfanfare.Bfamilie,BfamiliarizeBfame:Bfame,BfalteredBfalter,Bfake.B
faithfullyBfair,Bfailing,BfadedBfactoredB
fabricatedBfabioBeye-poppingBeychelle-baedBexzocoinB
extortion,BextortedBexponential.Bexpoed.Bexpoed,Bexploration.BexpiringBexpirieBexpired.BexpiredB	expertie.BexpaniveB
exhibitingBexhibit,BexhautedBexertedB
execution.B
excluivityB
exchanged.B
exchange).Bexamination,BexahaheBex,BewarBevolved.Bevolve.Bevilla,Bevil.B	evidentlyBevictionBeverityB	ever-moreBever-increaingBever-growingBever-changingBeventfulBeuropol.Beur.Betup.Betonia.Betn,B	ethiopianBethairBervice?BervantBerraticBernetBerdoBerc721Berc20,Berbia,Bera:B	equitableBequalityBepnBepionageB	epicenterB
epecially,BepaymentBepBeoul,BenvironmentallyBentrepreneurhipB	entitled:BentitleBenterprie-gradeBent.BenrollBenjoy.BenjBenglih.Benglih-peakingBengBenergy-richBenemieBenegal,BenegalB	enduranceB
endowment.Bendorement.B
endangeredBendangerBencroachingB
encompaingBencompaBenclaveB
encapulateBenabled.BemyonBemurgoBempowermentBemployment.BemphaticallyBemphaizing:BemperorBemmyBemmer,BemmaBeminentBeminenceBeminem,BeminalBemiion.BemiB
emergency.BembedB
elwartowkiB
ellithorpeBelliottBeller.Beller,Belki,BelicitedBelf-utainingBelf-regulation,Belf-directedB	elevatingB	eletropayBelephantBelement,Belectronic.Bele,BelderlyBelaboratingB	ekaterinaBeized,BeimicBeightie,BehBegypt,B	egwit-2mbBegg,BefficiencieB
effective,BefBeemaBeem.BeeleBeedingBeed,Bee,BeduardoB
editorial,Beditor-in-chiefBeditor,BeditB	edinburghBedge,BeddieBection,B
ecretariatBecoBeclipingBeclipeBecdaB	ecalationBebitBebay.Bebang.Beattle,B
earthquakeBearn.Bearly,Bealed,Be11evenBe-portBe-naira.Be-naira,Be-commerce,Be-cnyBe-blockBdynamoBdwarfBdurov,BduqueneBdupeBduffy,BduffyBdue,BdtBdryjaBdropped.Bdriver,BdrivenetB
dreyzehnerBdrearyBdreamrBdrawbackBdragnetBdownloadableBdownizeBdoveyB
doublelineBdot-comBdorey.BdominionBdominicB	dominate.Bdollar-backedBdoleBdogg,Bdogecoin-themedB
dodd-frankBdocumentation.Bdocumentation,Bdocumentary,Bdoctor,BdmitriyBdmitriiBdk,B	dividend.B	dividend,Bdivere,BdivedBdive.BditortBditcheBdirect,B	dipoitionBdiplomaBdipereB	dipatchedB	diolutionBdiluteBdilikedB
digruntledB
digitizingBdigibyteBdigB	differingBdifferently.BdifferentiatingBdifference,BdiehlBdicrepancie.B
dicovered.BdicoureBdicouragingBdicount,BdiciplinaryBdice.Bdice,BdiatifactionB
diapprovalBdiappointingBdianeBdiagnoedBdiadvantageBdiabled.Bdevil:BdeviingBdeviatedBdevatateBdetoken.B	detected,Bdeputy,Bdeputie,Bdepreciation,B
depoitory.BdepletedB	depictingB
dependencyBdentit,BdemontratorB	democrat,BdeluionBdeloB	deliting,BdeliberatelyBdelhi.B
delhi-baedBdektop,Bdeire.Bdegree.BdefyingBdefiedBdefieBdeficit.BdeficiencieBdefi-relatedBdefenderB
defendant.BdefconBdeertBdeepenBdeep,BdecryptionaryB
decriptiveBdecodeBdeckBdecide.BdechartBdecentraland,BdeceiveBdebutingBdebunkBdebate:Bdealing,BdeaiB	deadline,Bdead,BddBdcepBdc.Bdavo,Bdavidon,BdavidoBdaughterBdata-drivenBdarkerBdannyBdancingBdahlaneBdaemonBd83eBd.c.,BcyruBcyclingBcycle,B
cyberpace,B	cyberghotBcybercriminal.Bcutomer)Bcurrie,B
curriculumBcuratingBcuraBcunliffeBculprit.Bculianu,BcuhionBcryptyB
cryptovereBcryptotip.orgBcryptography,Bcryptocurrency-themedBcryptocurrency-poweredB
cryptocribBcryptocompare.BcryptocelebritieB	cryptobizBcryptoanarchitB
cryptoaet.Bcrypto;Bcrypto-worldBcrypto-proponentBcrypto-pecificBcrypto-paymentBcrypto-linkedBcrypto-fundedBcryonicBcryBcruzB	crumblingBcruhBcroatia,Bcritic.B	criteria,BcriptanBcriminality.BcrimeanBcrii-trickenBcrie.B	creightonB	creenaverB
creativityBcrazineB	cratchingBcraiglitBcovertBcovered.B
courtroom.BcouringBcoureyBcouple,Bcounty.Bcounterpart,B
counteringBcounterfeitingBcounelorBcott,Bcotland,BcotarB	correpondBcorrelation,B
cornucopiaBcorner,BcopyingBcoppayBcopie.BcoopedBcookB	converingBconvergenceBconvenience,BconvenedBconultedBconultativeBconultation.BconultBcontroverie.Bcontributor.Bcontribution.BcontitutingB
continued.Bcontet,B
contentionB
contender.BconnollyB	connectorBconnectivity.Bconnecticut,Bconideration,B
congreman,BcongoB
congetion,Bconfued.BconfrontB	conflict,B
confirmed:Bconfirm.B	configureBconfidentiallyBconfidentialityBconfidence.Bconervative.Bconduct.B
concludingBconcealmentBcomputeB	compreionBcomply.B
complimentBcomplieB
compliant.Bcomplexity.Bcompletion,BcomplementaryBcompetitive,B	competentBcompenatingB
comparion.B	companionBcompaB	communiquBcommentator,BcommendBcomitBcomic,BcomeyBcometicB	comedian,BcombingB	columbia,Bcolor.Bcolor,BcolluionBcolliderBcollateral,B	collapingBcollaboration,BcoldcardB	coinwitchBcoinwapB	cointext.B	coinrail,BcoinpiceB
coinource,BcoinmineB
coinmetricBcoingateBcoingappB	coinflex,Bcoineed.B
coindance,Bcoinatmradar.comBcoinatmradar,BcoinatmradarB
coinagendaBcoin?Bcoin.ph.Bcohen,BcofferBcoerciveBcoatue,BcoattailBcoaterBcoat.B	co-workerBco-leadBco-directorBcnmvBcmtBcme.BclubhoueBcloure,Bcloing,Bcloer.B	climbing.Bclimb.BclimaticBclimate.B	clienteleBclericB	clemency.B	clearnet,BcleanupBclark,Bclarity.B
clarified.BclaireBclaimed.BclaicalBclae,BclB	civilizedBcivil,B	city-tateB
circulate.BcirclingBciphertrace.BcipherBcionBcio,BcinnoberBcindyBcibBci-fiBchugBchromiumBchrome,BchipholBchip)Bchinee,Bchina:BchfBchewingBcher,Bchen,BcheddarBcheckoutBcheaper.Bchat,B	charitie,Bchargeback.Bcharacteritic,B
channelingB	changing,B
changchun,B
chancellorBchampioningBchamber.B
chairwomanB	chairman.B
chainpointBchadBcftc-regulatedBcftc,BcfeBcertifyBcertification.Bcertificate.Bcertain,Bcentralized.Bcenorhip-reitanceB	cementingBcementB	cellphoneBcelebritie.BcelebBceilingBcdBccBcaution.BcaucuBcaualtieBcatillo,B
categorizeBcartel,B	carrefourB	carolina,BcaroleBcarneyBcaringBcareyB
capitalim.BcapitaliationBcapita,BcapitaB	capacitieBcanyonB
candidate.B	cancelledBcampoBcamming,Bcammed.BcalviBcalled:BcalifornianBcalculator,BcalamityBcakeBcajee,BcahwebB
cahhuffle.BcahaddreBcahaccount.infoBcah-idBcah)B	cafeteriaBcadreBcabinet.BcabbageBbybit,BbuyucoinBbuy?BbuttcoinBbutingBbutaBburt,BburrellBburned.BburiedB	burglarieBbureaucraticBbureaucrat.BbundledB	bulgaria,Bbuild,B
buckminterB	buccaneerBbubblingBbubble?BbtcxBbtcpayBbtcbBbtc/eurB
btc-peggedBbruel.BbrookerB
broadeningB	broadbandBbritoBbring,BbrightlyBbrigadeB	briefing,Bbridge,BbribeBbreedtB	breathingB	breakout.B	breakawayBbravo,BbraggedBboydBbowedBboutB
boundarie.Bbotwana,BbottomedBboton-headquarteredBbotnetBbooterBboonBbomberBbolonaroBbologicBbogota,BbogatyyBbofaBboaBbo,Bbnb.BbmwBbmoBbm1397B
bloomberg:B	blockworkB
blocktack,B	blockpre,B	blockfyreBblockchain.info,B
blockbuterB
block.one,BblinkBbleepingBblazerB	blackout.BblackmailerB
blackjack,BbkBbitzlatoB	bitpoint.B	bitpoint,Bbitpea,BbitpeaBbitoodaBbitnzB	bitnomialBbitiraBbitinfochart,BbithareBbitgo.BbitekBbitcointreaurie.org.Bbitcointreaurie.orgBbitcointalk.org,Bbitcoinpaperwallet.comBbitcoinmap.cahBbitcoin.com.auBbitcoin-likeBbitbank,BbitbankBbitampBbirth.Bbip-70BbiotechBbinance?B	binance.uBbimuthBbilledB
billboard,B	bilateralBbigger.Bbigger,B	bifurcateBbicycleBbicuitB	bickeringB	bi-annualB	bharatiyaBbetter-than-expectedBbeta.B	bet-knownBbernieBbern,BbentoB
bengaluru,BbeneficiaryB
benefactorB
benchmark.BbelfortBbehaveBbegrudginglyB	beginner.Bbegan:BbedBbecome.Bbech32BbeatzBbeatleBbearabkyBbdnBbdcBbcvBbchn,Bbch/udcBbch/btcB
bch-focuedBbcah,BbcaBbc,BbazaarBbaumannBbat,BbarringBbarrickBbarrel,BbarkleyBbargain.Bbanner.BbankmanBbanker.B
bancoetadoBbanalBballoonBbalajiBbakinBbakerBbaitBbail.BbaiduBbahtBbahrain,BbahamaBbaggedBbafin.BbaffledBbaerBbaed.Bbadly,Bbadgerwallet.cahBbackwardBbackpageBbacklog,Bbacking,Bbacker,B	backdrop,Bback!BbachelorBb2bBaztecBazBaxoBaxieinfinity.comBawokenB	awakeningBawakeBavoided.B	avoidanceBavnetB	aviv-baedBaviv,Bavenue.BavaxBavalanche-baedBautracBauthorization,B	authoriedBauthoriationBauthor.Bauthentication,BauruBauringBaupiceB	auditing.BattributableBattractiveneB	attitude,B	attended.Battempt,BattainedB	attacker.B	atronaut,BatonB	atomicpayBatlaBatified.BatiBathleticBatanaBartifactBarretingBarockBarnoldB	armtrong.Barchitecture.B	arbitrum.B
arbitratorBarabia.Bapy.BaptlyBaptBapproximateBappropriate,B	appellateBappear,B	apparent.B	apparatu.Bapp).BapollonB	apocalypeBapifinyBapiece.Bapart.B	anywhere.B	anywhere,Banyway.BanxiouBanuragBanto,BantiwarBantitrutBanti-governmentB	anonymizeBannoyingBannaBannBankeiBangrilyBangle.BangleBangeredBandreyBandreen.BandradeBanctuaryB
anchain.aiBanarcho-capitalit,BanalyeBanalogueBanaBamyBamuingB	amterdam.BamritaBamnetyBammouBamitabhBameritrade,Bame-dayB	altcoinerBalphabetB	alphabay,BalpariBaloudBalong.Balo.BalmightyBalludeB
allegianceBallanBall:Ball-time-high.Ball-inBalive?BaliceBalexiBalert.B	alcoholicBalarie.Balabama,BakakovBak:Baitance.BairdroppingBaimeBailingBailBaifBaierBaienBaia-pacificBahujaBahmadiBahead,Bagriculture,BagoritB	aggrievedBaga.Baga,B
aftermath.Bafloat.Bafghanitan,BafghanBaffluentBaffirmBafemoonBaertion,Baerted.Baement.Badvied.Badvertiing,B
advantage,Badopted.Badopted,B	admitted.B
admittanceBadmin,B
adjutment,BadjutingBadjacentB	adherenceB	addition.B	addictiveBadd.BadamantBactimizeBact)BacrimonyBackerBacinqBachilleB
ach-backedB
accuation,Baccountant,BaccountableBaccountability,BaccordB
accomplih.B	acceible.BabuzzBaburd,BabuiveBabruptBabout,BaboundBaboardB	abilitie.BabandonBabahBaayog,Ba16z,Ba16zBa1B[cryptoB[btc]B[are]B=B8amB79B76,000B729B70.B6nmB69,370B6173,B6.8B6.3B56,000B550B55,000B544B540B5.4B494784.B460B45thB45.B40,824B4.4B375B37,B36,000B35thB35-year-oldB34,B332B320B300kbB30-year-oldB30-yearB30-40B3.2B3.0.B3-15B3,813B3,000+B2dB2:10B280B28.6B27,000B262B26,000B25th.B25-year-oldB24thB240B230,000B23-year-oldB2060B205B2033B2030B2026orB2026andB2025,B2025B2021-22B201dpartnerhipB201d?B201d).B201cye,B201cwildB201cwereB	201cwell,B201cwelcomeB
201cwealthB201cweakB
201cwarnedB201cupB	201cuntilB201cuchB201cubtantialB201ctrutB	201ctotalB	201ctigerB201cteveB
201ctetherB
201ctartedB201ctartB201crunningB201crequireB201crepreentB201creponibleB201cregulatedB201creaonableB201cratB201crapidlyB
201cpurelyB201cpuhB201cpropoedB	201cproofB201cprohibitionB201cprivacyB201cpreparedB201cportraitB201cpompB	201cpleaeB201cpentB201cpeculativeB201cpecificB	201cpanicB201covereignB201coldB
201cnoticeB201cmutB
201cmovingB	201cmoralB201cmoderatelyB201cmodelingB	201cmaybeB201cmatermindB201cmallB201clookingB201clookB201cliveB201clitB201cliquidityB
201clikelyB	201clightB201ckeyB201ckeepB201cinvetigationB201cinitialB201cincreaedB201cillegallyB201cidentifiedB201chypeB201choldB201charmfulB201chareB201cguaranteedB201cgrowingB201cgreaterB201cgraduallyB201cgettingB201cgenuineB
201cgalaxyB201cfundB	201cfoundB201cforeignB201cfindingB201cfewB201cfearB201cfacilitateB201cexploringB201cexploreB201cexploitB201ceveryday:B	201centerB201cenormouB201cengagedB201celB201ceentiallyB201ceconomicallyB	201ceaierB201cdynamicB
201cduringB201cdogeB201cdefinitelyB201cdeepB	201cdeathB201cdeadB201ccryptocurrency,B
201ccreateB201cconideringB201cconcernB201ccomputerB201ccomplianceB201ccommunityB201ccombatingB
201cclientB
201cchangeB201cceaeB
201ccannotB
201cbullihB201cbullB201cbtcB	201cbringB201cbondB
201cbetterB201cbanB201cbalancedB201cbabyB	201camongB	201calwayB201caltcoinB201calreadyB201callowingB201caleB201cactivelyB201cactB201caccelerateB201caboluteB	2019haganB2018trutB2018thiB2018richB2018regulatedB2018redB2018pednB2018operationB2018nextB2018longB2018fairB2018electricityB	2018cahedB2018buryB
2018bondedB2018bB2014andB2003.B2003,B2003B20-year-oldB20-yearB2.9B2-8B2-15B1gbB19th,B1994.B1991B1987,B1984B1981.B1971,B194,775B1933.B19.5B19.4B183B18,000B17,000B160,000B16,796B144,336B140,000B14.5B14.4B130,000B12:37B121B12.3B12-nightB119,756B115,000B113B110+B10:30B101xB1000xB100-dayB100,B100+B10.7B10.4B10-11B
10,000,000B1.7mB1,500+B0ba9B0b8eB0639B062fB0438B0422B00f6rgB00f3,B00f1oB00edn,B00edguezB00ed,B00e4rdexB00a35B00a34B00a3200B00a3114B00a0zeronetB00a0xapoB	00a0wouldB00a0wiB	00a0whichB00a0wereB00a0weiB	00a0wedihB	00a0wedenB
00a0walletB00a0vouchingB00a0ventureB00a0uerB
00a0twitchB	00a0threeB00a0thouandB	00a0thereB
00a0tetingB00a0tatementB00a0tablecoinB00a0rolloutB
00a0rippleB00a0reearcherB00a0rareB
00a0rapperB00a0pvB00a0propertieB00a0privacy-centricB00a0preidentB00a0pbocB	00a0panelB00a0otherwieB00a0oB00a0nowB00a0norwegianB00a0nickB
00a0nchainB	00a0mocowB	00a0minerB00a0mayB
00a0largetB00a0laB
00a0krakenB00a0kimB00a0jutB00a0jpB
00a0jonaldB00a0iranianB00a0irB00a0individualB00a0ifB00a0heB
00a0georgeB00a0freeB	00a0fiftyB00a0euB	00a0eightB00a0economicB	00a0ebangB00a0cutomerB00a0cheduledB
00a0caymanB00a0cahB00a0btcB00a0breadwalletB00a0bitfuryB00a0bitflyerB00a0binanceB00a0bigB00a0berbankB
00a0belaruB00a0antonopouloB00a030B00a010B0.8B0.28B0.15B0.03B0-confB/r/btcB.netB.999B	**update:B(ze)B(xem),B(wp)B(wef)B(virtualB(va)B
(unacrip),B(um)B(udh).B(ton)B(to).B(toB(tmc)B(roughlyB(ro)B(pwc)B(pud)B(pt).B(payB(pax)B(pac).B(pac)B(ov).B(otcmkt:B(ofB(occ).B(occ),B(nydig)B(nydf).B(nya)B(nowB(nm)B(nirp).B(mma)B(mlb)B(miota),B(looelyB(link)B(le:B(kwh),B(knowB(it)B(itB(imc)B(htp)B(gdax)B(foia)B(finra),B(fdic)B(fca).B(f),B
(europol),B(etp).B(eo)B	(egwit2x)B(ed)B(doj).B(dnb),B(dn)B(dcg),B(dapp),B(dao)B(ctor)B(cma)B(cia),B(cex)B(cbb)B(care)B(bu),B(btr)B(btcp)B(bito)B	(bitcoin)B(bia).B(batm)B(bancoB(ba)B(autrac)B(ar),B(amld5)B(aicB(ai)B(ada).B(@100trillionud)B(4),B(4B(22),B($1.3B$900mB$9.8B$9.6B$9,100B$8.4B$760B$700kB$700,000B$7,900B$7,600B$7,500B$69.3B$66B$64B$630B$63B$625B$62B$61B$600k.B$6.98B$6.9B$6.7B$6.6B$6,200B$55,000B$54kB$534B$52.5B$51kB$50mB$5.3B$5,600B$49B$48kB$4800B$46B$45B$43kB$43B$429B$402B$400mB$4.7B$4.4B$4,200B$381B$32kB$31B$300bnB$3.85B$3,900B$3,700.B$2bB$29,300B$281B$277B$2700B$2300B$23,777B$225B$2.8B$2.06B$2,500B$1bnB$19.5B$19,000B$187B$182.5B$182B$1600B$15,000.B$148B$145B$142B$132B$127B$125B$117B$110mnB$106B$100k.B$100bB	$100,000,B$10.4B$1.75B$1.26B$1,700B$1,400B$0.50B$0.30B$0.20.B$0.03B$0.01B$0.006Bzurich.Bzuckerberg,BzonteBzoningBzk-narkBzkBzirlinB	zimdollarBzigluBzhou,BzhavoronkovBzeuBzerolinkBzero-umBzerB	zenminingBzclaic,BzapBzanderBzamBzackBzacharyBz9Bz.comByunbi,Byu,ByuBytem;ByouthfulByour.ByoungterByobitByntaxBynergieB	yndicatedB	yndicate.B	yndicate,B
ynchronizeBynapeB
ympatheticB
ymmetricalBymbioticBylvainByichuanByfdexB
yen-peggedByellen.Byearly.Byear:B	year-end,B
ydney-baedBycoinByapizonByankeeByandexByanchengByamadaBxrb,BxmaBxiaomiBxiaolin,BxiaolaiBxhoneybadgerBxfaiBxetra.BxetraBxcpcBxapo.BxangleBwyden,BwweBwrite:Bwrite,BwretlingBwreakingBwrayBwrath.Bwp-baedBwpBwouldveBwould-beBworthle,Bworth,Bworry,Bworld?Bworld-leadingB
workforce.Bworker.Bworked.Bworen,BwordedBwoop.BwongBwolf.betBwojak,Bwoe.BwiveBwitne.B
withdrawn,BwithcoinBwitcheBwire,Bwinner,Bwinkelmann,BwingingBwindfallB	wimeijer,BwillfulBwilfullyBwildly.Bwildly,B	wildfire.BwildetBwilderneBwild,Bwie,B
widepread,BwidenedBwidely.BwickBwhom,B
whitetreamBwhite-labelBwhimicalBwhere,BwhcB
whatoever,B
whale-izedBwetpacB	weternmotBwenzhou,BwenhengBwenceBweltoB
wellingtonB
well-wiherBwell-meaningBwelcome.Bwelcome!Bweird.Bweigh-inBweibo,B
weexchangeBweet.Bweet,Bweekend?Bwedbank,BwedbankBwechat.BweburyBwebcatBweatBwealthy,Bweaknee.BwavingBwaveringBwaton,B	watching.BwatchfulBwarmlyBwarhipBwarhBwarfare,Bwaraw,BwarawBwanted,Bwanted!BwaneBwan,BwalletconnectBwallet?BwallaBwaldron,Bwake-upBwagingBwaggingBwager.Bwager,Bwage,Bwag,BwagBw3cBvullo,BvulloB
vulcanvereBvtcBvroman.BvpeBvoyager.Bvoucher.Bvoting,BvontobelBvoluntaryim,B
voluntary,Bvoluntarily.Bvolcano.BvoiineBvoid.Bvoice,BvogueBvodafoneBvizcayaBvirtuaB	virginia,Bviral,Bviolent,B	vinnytia.BvillalbaBvillagerBvillage.Bvillage,Bviitor,Bviit.B	vigoroulyB	vigilanteB	vigilanceBviewing.Bvienna,BvidyaBvictor.BviatBviacoin.Bviabtc.BvevueBvery,BvernBvermont,B	verified.Bverge.BveraityBverabankBventure-capitalB	venerableBvelocityBveklerB	vega-baedBvega,Bveene,BveelBvector.BveBvc.BvauldBvargaB	vanitygenB
vanderwiltBvalidation.Bvalidation,BvalenzBvalenokB	valencia,BvalenBvailiev,BvaguelyBvacuumBvaccinationBv2.Bv-poolB
uzbekitan.BuwonButteringButteredButorgButilizationButaBurvive.Burvival.Burveyed,B
urveillingB	uruguayanB
urrenderedBurrenderBurnameBurgicalBurgeryBurgency.BuretyBurbtc,B	urbitcoinBupward.Bupward,Buptime,BuptBuppreingBupply-chainBupplie,BuploaderBupload.Bupicion.BupiBupheavalB	upgraded.Bupertar,B	upermodelBupermarket,Buperman,B	uperheroeB	uperedingB
uperchargeBupenion,Bupended.Bupdrive.Bupdated.B	unwindingBunwillingneBunutainable.BunuedBunuableBuntrutworthyBuntrueBuntaxedBunregulated.BunquetioninglyBunprovenB	unpleaantBunoundBunophiticatedB
unolicitedB
unnoticed.B
unneceary.BunmovedB
unlinkableB	unleahingBuniveritie.B
univerallyBunique,B	unionpay,BunidoBunicorn,BunicodeBuniB	unhealthyB	unfoundedB	unforeeenBunfold.BuneayB	unearthedBundicoveredB	undertakeB
underminedBunderliningB
underlinedB	underlineBunderecretaryBundercollateralized.B
undeniablyB
undeirableBuncontrollableBuncontitutional,BunconditionallyBuncleB
unchecked,BuncannyBunavoidable.Bunavailable.BunaumingB
unambiguouBunairedBunaffected.B
unaffectedBunaccountedB	ummarizedBumitBumaBuluhuBultimately,BulmartBuitablyBuiani,Buhi,BuheringB	uggetion.B	uggetion,BuffolkBufficeBufc,Buername,BuefulneBudan,Bud/btcBuch.BuccumbBucceiveB
uccefully.Bucceed.BucahB
ubtitutionBubtcB	ubreddit.B
ubramanianBubpoena.BubmergedBubided,Bubcription,Bub.BuatpBuamiBuae-baedBu2fBu$2000Bu$1BtyrannyBtype:Btyle,Btwo-tierB
two-minuteBtwo-hourBtwitedB	twenty-ixBtweet:Btv18BtuurBtutanotaB	turnover,B
turnaroundBturbulence,B
turbulenceBtunnelBtuniiaBtuff,Btud.BtuckedBtubhubBtubeBtrying.Btry.BtrxBtrutvereBtrutle,Btruted,Btruglia,Btruggle,BtruefiBtrue:Btruck,BtrpB
troubling.BtrottedBtrophyBtropeB	tron-baedBtroll,BtrivialBtriple-digitBtrip,Btring.BtrimB	trillion)Btrick.BtribulationBtri),Btrezor?B	tretchingBtrend:BtremmelBtremendoulyBtreiBtrefulBtree:Btree.B	treaurer.Btreaure,BtreatieB
treamlinedBtreamlabBtream.BtreakBtre-tetB
trc20-baedBtrayB	traveler,Btravel,Btravala.com,BtrappedB
tranportedB	tranport,B
tranpired.B
tranmittedBtranlated):B	tranitoryBtranitionedBtranformer,BtranformationalBtranacting.Btranacting,Btranact.Btranact,BtrajectorieBtrail.BtrahBtragicBtragedyB
traffickedBtraffic,Btrader?Btraceability,Btrace.BtoxicB
townville,Btournament,Btournament!Btourim,Btour:BtotleB	totallingBtotalitarianBtory?Btory:BtorridBtoronkyBtormyBtormx.Btorm,BtoreholdBtopp,Btop-upBtop-rankingB	top-notchBtoomey.BtonkohkurovBtonightBtomorrowBtommyBtomatoeBtokenpayB	tokenomicB
tokenized,Btokenization.BtokenaleBtoken]Btoken:B	tockholm.B	tockholm,B
tockholderBtnBtmzBtmc.Btlaib,BtitovBtitanicB
tipulationB	tipulatedBtipbitcoin.cahBtinghuaBtingBtimurB	timulatedB
timeticketBtimerBtimehareBtime),BtileBtiktok,B	tightenedBtifledBtidyBtidalBtickmillB	ticketflyBtickedBtick,BticalBthurday)BthunderBthrottleBthrone.BthrivedBthrive,Bthree.B
three-weekBthree-judgeBthought,B
thorchain,Bthirty-fiveBthirty-evenB	thinking,B
think-tankBthief,Bthiam,Btheymo,BthereinBthereafter.BthereaBtheorit,Btheoretically,Btheme.Btheme,Bthem?Bthe]Bthe9,Bthat?BtharmanB	tezotopiaBtezo.Btext,B	tetimony,B	tetimonieBteted.Bteted,BtetaBterriblyBteroid.Btern,BterminologyB	terminal,BterceraBtenion,BtengeBteng,BtenetBtendoBtendingBtellarx,B	telephoneBteinBteheranBteemit,Bteel:B	tecracoinBtechnology?Btechnological.Btechnoking,Btear,BteaerBteadily.Btb,Btay,BtaxmanBtatyBtatute.Btatita,B	tatement:BtatemanBtate;B
tate-levelB	tate-baedBtarwoodBtarvingBtart-up.BtarkwareBtark,Btariff.BtargroupBtarbuck,BtapeBtantalizingBtank.BtandardizedBtandardizationBtand,B	tamperingBtamper-proofBtamil,Btalking.B	talk-baedBtalibanBtalentedBtakerB	takedown.BtailBtahlBtagnatedBtaggedBtadium,BtackledBtackedBtack.Btack,BtabledBta.BryverBrunnerBrungtedBrunetBrumor.Bruler,BrulerB
rulemakingBruian-peakingBruian-languageB
ruian-baedBruh.Bruh,BruffledBruchirBrubin,B
rubentein,BrubBrrBrpgBroutedBrout,Brouhani,Broubini)BrotovBrotinB	rothchildB	roottock,BroottockBrookieBronnieBronaldBrome,BromeBrollout,BroljicBroilandBrohdeB
rogozinki,BroganBrofinmonitoring,Broe,BrockingBrocketrB	rocketingBrobut.Brobo-advierBrobleBrobinon,B
robinhood.B	roberton,BrobbingBrobbinBrobberyBrobbBroamingBroadideBrmbBrizun,Briver.Brival,B	rippeningBripioBriot.BrinkBringingBring,BrikingBrik-baedB	rigoroulyB
rightfullyB
right-wingB
right-handB	ridiculedBridge,Bride.Brico.Brico,Brice,BriceBricanBrica.Brica,BricaBricBribeiroBribeBrial,B
rhetoricalBrexmlBrevolutionizingBreviitedBrevertBreverberateBrevealerB	revampingBretweetB	returned.B
retructureBretreatB	retracingBretiringB
rethinkingB	requeted.B	repreent.Breponibility.B	reponded,B
repoitory.Brepo,Breply.BrepayingBrepatriatedBrepaidB
reorganizeBreopenedBreolvingBrent.BrenoB	renminbi,BrenegingBrenbtcBrenameBremuneration.B	remotely.B	remitano,BremedieBrelmBreliveB	relinquihBrelihB	religion,B	relevant,BrelayedB	relative,BreinventingB	reimburedB	reidence.B	reidence,BreidedB	regulate.BregitrarBregencyB	regardle.BregalBrefutingB	refundingBrefued,B	refinerieBreet.BreedB	redundantB
reduction,BredirectingB
redirectedB	redeemingBredeeemBreddyBrecycleBrecuedB
recountingB	recountedBreconnectionB	reconiderB
reclaimingB	reckoningBreckonedBreceipt,BrecapB	recalled.B
rebuildingBrebound.Brebound,BrebornB	rebellionBrebalancingBreauredB	reauranceB	realized,BrealeB
real-time,BreaignedBreadyingB	readerhipBreadableBread:theB	reaction.B	re-openedBre-enterBrbz,Brbc,B	ravikant,BraviBratliffBratio,BraphaelBrapberryBrantBranom,BrandomneBrand.BranadivBramirezBraid,BraghuramBragedBraffleBracoffBrackingBracketBr/millionairemakerBr/cryptocurrencyB
r/bitcoin,BquirkyBquietly,Bquib.Bquerie.B
queenland.Bquarter-on-quarter,BquarrelBquared.Bquality,BqualitieBqualitativeB
quadrupledBquadB	quabblingBqtum,BqinghaiBqiBqe.Bq)Bpyramid.B
pychology:B
pychologitBpuruit,BpurpoefullyB	purpoefulBpurpoe:BpurpleBpuriouBpuriBpurdyB
purchaing,BpunterB	punihmentB	punihableBpune,BpundiBpuncheBpunBpuh.BpufferB
publihing,Bpublication:B
publicallyB
public-keyBpubgBpuBptdBptBprudenceB	provided.Bprovide,Bproven.B
protractedB
prototype.B
prototype,Bprotitution.BprotitutionBprotetationB
protected.Bproportion.BprophetBproper,B
propagatedBproof-of-take.B
pronouncedBpromulgatedBprompt.cah,BproliferatingBproliferatedB	progreionB	progreed,Bprogre?BprogpowB
profundityBprofitable,Bprofeor:BprofeingBprofeedBprofeeB
proecutor.B	producer.BproclaimingBproclaimB	proceing,Bprocee,B
proceduralBprixBprivyBprivacy-invaiveBprivacy-enhancedBprivacy-conciouBpripachkin,B
pripachkinB	printableB
principle,Bprince,B	primitiveBpricing.Bprice)Bprevention.Bprevail.Bpretty.BpretextBpreponderanceBpreparation,BpreoccupiedBpremineBpremie,B	preloadedB	preident:BpreidedBpreference,Bprefecture.Bprefecture,BpreervationBpreent:B
preeminentBpredictably,Bpredict.Bpredicament.B	predatoryBprecriptionB	precribedBpreconditionBprealeBpreageBpre-taxBpre-regiterB
pre-loadedBpraxiBprakahBprague-baedB	practicedBpowwapB
powerhoue.B	powerful,B	powerballBpoueBpotponementB	potponed.B	potlight,BpotholeBpot-warB	portrayalB
portmouth,Bport.BporeB	populace.Bpop-cultureBpop,Bpooling,BpoolingBpoolin,BpoolerBpontaneoulyBponor.B
pompliano.BpompeoBpolygon.B
polychain,B
pollution.B	pollutionBpolluteBpoll.Bpolker.gameB	polkadot,BpoliticoBpolitician:Bpolitic.Bpolicy?B	policemenBpolice:Bpoli,Bpoibilitie,Bpoeion,B	podcater,B	pocketingBpocket,B	pnetwork,BplutuBplunge.Bplummeting,BpluggedBplight,BpleyerB	playthingBplayaBplauibleBplatzerB	platinum,BplateauBplateBplanner.Bplanner,Bplanet!B
plaintiff.Bpizza,BpitcheBpitch,BpitBpiratetokenBpiratedBpiral.B	pipeline.Bpioneer,BpimcoBpillingB	pilipina,BpilferedBpiked.BpiixpayB
picturequeBpickleBphyicBphotovoltaicBphotohoppedBphotohopBphoto,Bphone:Bphoenix.Bphilanthropy.B
phenomena.Bpeudonymou,Bpeudonymity,B	peudonym,BpetyaB	petroleumBpervereBpervadedBperuingB	peruadingB
perth-baedBperthBpertainBpermiionle.B	permiion,BpermeateBperkinB
periodicalB
performer,B
performed.B
perecutionBpepperBpeople?BpennBpenetrationB	penetrateBpendbch,BpendalBpendableB	penalizedBpenalizeBpeltBpekingB
pejorativeBpeedingBpeculative.Bpeculative,B	pecifyingBpeceBpeaking.Bpc,BpaywardBpayoneerBpayment?BpayerBpayafeBpay-per-viewBpaxforexB	pavithranBpaulo,Bpaued.BpatriBpatorBpatimeBpathwayBparteBparlorB
parkprofitBparent,Bpare.BparanoiaB	paragraphBparadi-royerBpanther.Bpantera,BpankinBpanjerBpanigirtzoglouBpanickedBpanic.BpangeaBpanewBpanellitBpanel:Bpanel.Bpanel,BpandoraBpancakewap.Bpancakewap,BpammingBpalihapitiya,BpaleBpakBpairing,B	painting,BpaintakinglyBpai,Bpage:BpacunBpackagedB	pacecraftBp.m.,Bovr.aiBovex,BovertookBovertokenizationB	overthrowB
overtakingBoverruleB	overhaul.Boverea,B
over-hypedBoutwitBoutweighBoutlet:B	outlandihB
outhweternB	outhboundBoutflow,Boutfit.Boutfit,B
ourceforgeB	oupernerdBoundingBoulBoto,Botc.Bortiz,BortingBortedBorionB	originateB	original,BorientationBoregon.Boregon,BordealBopulouB	optimizedBoptimal,BoptimalBopra),BoppreiveB	opponent,BophiticationB	operated,Bopened.Bopenbazaar.Bopec.Bop_mul,Bop_code,Bop_checkdataigverifyBop_checkdataig.B	op-returnBop-ed,Bop-codeBony,B
onnenhein,BoniteBongoing,Bone-manB	one-clickBonchain,Bon-ramp.BomnibuBomiionBomiego,Bomg,BomenicBomeday.Bomeday,BombudmanBomarBolvableBolivierBolidity,BolidityBolegBoldierB	olaribankB	oklahoma.Bokay,Bokamoto,BohB	offloadedB	official:BoffhootBoffence.BoffenceBof:BoedBoecdBoddlyBodditieBodaBoctBocietie.BocietBocialiteBocar-winningBocaio-cortezB	obviouly,B
obtructionBobtain.Bobolete.BoblivionBobligeBoblatB	objectingBobject.Bobfucation.Boberver.Boberver,BobelikB	obanikoroBoathBoaring.Boared.Boar.BnwaniobiBnuzziBnuggetBnubank,BnrccBnowden:B	novoibirkBnovi,BnovelitBnovel,BnounBnotwithtanding,B	notifyingBnotifieBnoticed,BnotalgiaBnorway-headquarteredBnortheatB	normally.B	normalizeBnoopingB
nonpartianBnonfungible.com,B	nonenicalBnoncutodial,Bnon-u..Bnon-topBnon-reidentBnon-malleable.Bnon-financialBnon-europeanBnon-employeeBnon-deliverableBnon-cooperativeBnon-compliantBnon-cahBnominateBnoiyBnoireBnoie.cahBnoedive.Bnode?Bnode40,Bno2xBno-noBnitorBnirpBninjaBnine.Bnimbu,Bnikkei.B	nightlifeBnickelodeonBnicelyBnguyen,Bngo,BnexiBnewagentBnew;BnevanoBneutral,BneufundBneu-nerBnetworking.Bnetworking,BnetorBnetoBnetellerBnetcoinB	net-worthBnem.B	neighbor.Bnegotiation,B	negotiateB	neglectedB
negativityBneedntBnedvedBnedBneckerBneckB	neceitie.BneceitieBneatBnear.BndlovuBncrBnciBncaBnbcBnavy,BnavalnyBnaughtyBnatoBnative,B
nationallyB
nationalitB	national.BnaryBnappaBnaphot,B	nanometerBnameleB	namecoin,BnamecoinBnamebaeBname)BnaiveBnaira,BnairBnagarBnaerBnacionalBnaauBna:Bna.Bn411BmythicalBmyth.B
myterioulyB	mycrypto,BmyceliumBmyanmarBmwedzi,BmvBmutual.Bmutual,BmutantBmut.BmurkyBmurder,BmurdaBmurcomB	multivereBmultiplyingB
multimediaB
multi-tageBmulti-purpoeBmulti-erviceB
multi-coinBmulti-blockchainB	muhammaduBmugglerBmuggingBmueum.Bmueum,BmudaBmudBmuclingBmuch-anticipatedBmucatBmt.goxBmoved,Bmove?BmottoBmotleyBmotive.Bmother,B	mortgage.BmortemBmorri,Bmoro,Bmorning:BmorelB	morehead,BmorallyBmootherBmootedBmoore,Bmooning.Bmonthly,Bmonth-on-monthB
month-longBmontero,Bmonter.B
montenegroB	monopoly.BmonopoliticBmonkeyB
monitored,B	monicker.BmoneywebBmoneyitBmonero-baedB	monacoin,BmonacoinBmonaco,B	molyneux,Bmoldova,BmoldingBmoldBmoeingB	modifyingBmoderation.BmobiBmoaicBmoB	mnemonic:Bmm,BmlBmix,BmiundertoodBmirepreentedB	mipellingBmiouri,Bminting.Bminted.Bminneapoli,BminivanBmining?Bmining-relatedB	miniatureBmingle,Bminer)BminceBmimblewimble,BmimanagementBmillennium,B
millenniumB	military,BmileyBmiledB
mileading,Bmildly.Bmilan,BmihapB	mifortuneBmieryBmied.BmidlandBmideaB
middlemen.B	middlemanBmiddle-agedBmid-november.B
mid-march.Bmid-february,Bmid-december.B
mid-augut,Bmid-afternoonB
microneianBmichief-makerBmichellBmhlangaBmfa.BmezaBmexico-baedB	metropoliB	metromileBmetcalfeBmetawarB
meta-trendBmerrickBmercyBmerchandie.Bmerch,BmercadolibreB
mentioned.Bmention.B	mentalityBmendoza,Bmemory,B	memorablyBmemoirBmemo:B	memo.cah.Bmeme:Bmeme-cryptoB
memberhip,BmeltBmelker,B	melbourneBmeingBmehnetBmegaway,Bmegaupload,BmegabankBmeet.BmeedBmeebitBmeduzaB	medicine.B	medicine,BmeccaB	meanwhileBmealyBmco2BmckelveyBmcelroy,B	mcdonaughBmccourt,BmccarthyBmccabeBmcadamBmbtc,BmazzucaBmayweather,BmayhemBmayer,B
maximalim,Bmax.BmaviB	maverick.BmaventaBmauritiuBmauB	maturity,BmaturityBmatured,BmathiaBmathematic,B
mathematicBmaterpayment,B
maternode.B
maternode,Bmaterialize.Bmaterialize,B	maryland,Bmarvel,B	martplaceBmartkeyBmarter,Bmart.BmarlinBmarku,BmarionBmarineB
marijuana,BmarginalizedBmarcheBmarcelBmarapoolBmar,Bmao,Bmanufacturing.Bmanufacturing,BmanouriBmanion,BmanifetationBmania.B	maneuver.B
mandatory.B	mancheterB
managerialBmanage,BmanaBmamutualBmammothBmalware.Bmalta.BmalleyBmall.BmalignBmalcolmBmakoriB	makeover,Bmak.Bmaire,BmaiquetB
maintream?B	mainland.BmainichiBmailedBmail-inBmaidenBmahome,BmahhadBmaheBmagnateBmaena,Bmadrid.Bmadrid-baedBmadionBmaddeninglyB	maddeningBmacron,BmackB
macedonia,BmacauBmaarten.Bm30Bm2Bm10Bm.payBm-peaBm-baedBlynnBluxfiB
lundeberg.Blumpur,B
luminarie.BluleaBlugeBluck.BlppBlowmitBlowdown.Blow-interetB
low-carbonBlove.BloungeBloui,BloudetBloud,BloudBlouBlord,BlopezBlonghahBlongfinBlonget-ervingBlonger,B	long-formBlogic.BlockwoodBlocked,BlockeBlocatingBlocally,Blocalethereum.Blocalbitcoin.com.BlobbanBlndBllmBlivelyBlive-treamedBlitigation.B
literally,BliteBlira.Bliquidator.Blippage.BlinkinB	linkedin.BlingerBlineup,BlinerBline:Bline-upBlindeyB	lindemannBlimitation.B
limelight.Blikewie,BlikeningBlikelietBliked,BlikBlightetBlightedBlifepan.Blide.BlickBliceneeBlicened.BlibreB
liberland.BliberiaB
liberatingBliao,B
liability,Bliabilitie,BlgoBlexBlevionBlevine,BlevineBleumiBletteredBletter:BlethalBleonteqBleoneBleninBlending.BlegitimizedBlegit.B
legilator,BlegilateBlegally.BleftoverBleft-leaningBleewayBleep.Bleary,Bleap.Bleak.BleaingBleaedBlcxBlbcc,Blbank.Blazy.comBlawleBlavicB
laundromatB
launchpad.B
launching,BlaudingBlatter,BlatingBlatet,B
lat-minuteBlarger.BlaoluBlankanBlamentedBlambo.Blambo,Blamau,Blam,BlakhBlake,Blaiez-faireBlaherBlahed.BlagBlabyrinthineBlabor.BkytBkyrocketing.B	kyrocket,Bkyiv.Bky-highBkwon,BkwachaBkviliBkuniBkunaBkudrinBkuaBkrugman,Bkruger,BkronerBkroneBkranovBkorverBkorobogatova,BkoobBkong-headquarteredBkoinex,BkohB	koeterichB	kobayahi.Bknow:B	knock-offBknifeBkneetBkirtBkirillBkipngBkipchogeBkingwayBkingdom-baedBkindlyBkin.Bkin,Bkilowatt-hour,BkilowattB	kilometerBkiller.Bkiller,Bkilled,Bkik,BkicktartingBkick-tartedBkick-offB
khorowhahiBkhannaBkhalilB	kganyago,BkeyportBkeyboardBkewedBkew.Bkew,B
kenya-baedBkennyBkennelBkennedy,Bkelman,BkelerB
kazakhtaniBkatyBkathyBkateBkarmaBkaprukaB	kaplikov,Bkanoon,BkangBkampala,BkamalaBkaliningradBkali,BkaiBkadyrovBkabulBk-7BjuxtapoitionBjutifieBjut.gameB	juridicalBjunctionBjumptartBjumped,BjulietteB
juggernautB
judiciary.B
judiciary,B	jpmorgan.BjpmBjoytreamBjourBjoltBjokeyBjoin.Bjohanneburg-baedBjoe,BjnkBjio,Bjinping,BjiBjewelry,BjetmanB
jeopardizeBjeffrey,Bje,BjavadBjargon,Bjapanee.BjaneiroBjamBjaitley,BjailingBjaideepB
jacquelineBjackpot,BjabBj.p.Bizvetia.BizvetiaBixth-largetBivoryBivanovBitbitBitauBitadakiBironicBirnaBiraqBira.Bira,Biphone:BipcBip,BiouBiolateB
invoicing.B	invoicingBinvetor:BinvetopediaBinvetigator.Binveted,Binvet),B	inventiveB	invented.BinvalidBinvaion.BinureB	inurance.BintuitB	intrigue,BintrigueBintotheblockBintinetBintimidatingBinterviewerB
intervene.Bintertwined.BintertwinedBinternet-relatedBinternationally,B
intereted,BinterdiciplinaryBinteraction,Binter-exchangeB
intenifie.B	intendingBintellichipBinteligenteB
integrity.Bintant.B	intalmentBinr.Binquiry.B
inpirationB	inolvent,B
innovator.Binnovative,Binning.BinmateBinjuryB
injection,B
initiated,BiniterBinit.BiniktBinide.Bingve,BingveBingularity.B
inglepointBingingBinfuriatingBinfuionB
infringingBinfringeBinfographicBinflow.B	inflictedBinflictB	inflated,B	infinity.Binfante,Binfancy,B
inexorablyBinertedBinequality.BinemaBindutry:BindulgeBinduceBindian,B
india-baedBindexerBindependent.Bindependence.B
indefiniteBindebtedB
incumbent.B
incubator,B
incubatingB	incubatedBincreaingly,BinconvenienceBinconitencieB
including,BinclinationBincitedBincidentally,BincheBincarcerationBincarceratedBinboxB
inadviableBinadequate.Bin-tore,Bin-fightingBin-builtB
in-brower,Bimultaneouly,B
imultaneouBimulateB	improved,BimproperB
improbableB	imprionedBimpractical.BimpoverihedBimpoterBimport,BimpoibilityB
implifyingB
implicity.B
implicity,BimplefxB	implecoinBimple:Bimple-to-ueBimpetuBimperative,BimpartBimmortalizedB	imminent,B	immigrantBimmerionBimilarweb.comBimdb,B	imbalanceB	imagined.Bimagine.B	imaginaryBilverblood,BillutrativeB
illutrate,BillneB	illicitlyBill.BilentlyBilencedBilence,B
iland-baedBiifBihaanBignup,BignotuBigned.Bignatov.B	ignatorieBignalledBignaBign.BightingBigamingBifwg,BiftedBifp.Bif,Bieo.BieBido,BidetepBideologyB	ideologieB
identifierBidentified.Bideline.Bidea?Bidea;B	iconoclatBico:Bice3x,Bice.Bice),BicapitalBicahnBibm,BiberiaBiabelBiaacBi?Bi2p,B
hyperbolicBhyper-inflatedBhydropower,Bhydro-quebec,Bhydro-quBhydra.B
hyderabad.BhvartBhuttonBhutterBhurt.BhurryingBhumorou,BhumboldtB	humankindB	humanity,Bhuman.BhulaBhughe,Bhuge,Bhuffle.BhuatieBhroudedBhriftB
hranilnicaBhowerBhow?BhovelBhouton,BhoutingBhoutB	hour-longBhould.B	houehold.BhotzBhotmine,BhotlineBhotageBhortly,BhortlitBhortfallBhortenBhopping.Bhopping,Bhopital,BhopelelyBhooverBhooterBhoopB
hoodwinkedBhood,Bhood)Bhonk,B	honeywellBhoneybadgerBhondura,BhonduraBhomele.Bhomele,Bholm.Bholm,BhollandBhole,B	holbertonBhoggingBhoffman.Bhoffman,BhofBhodling.BhockingBhock.BhobbieBho,BhmtBhitorically,Bhitcoin.Bhired,Bhire,Bhip.Bhip,Bhinman,BhinilBhingeBhine,BhilledBhikedB
hijacking.B	hijackingBhijackBhighlandBhigh-volumeB
high-valueBhigh-qualityBhigh-poweredBhigh-ecurityBhierarchicalBhieldedBhide.BhiddenwalletBhiatu.Bhezi,Bheyday.B	hevchenkoBheroinBhernBhermitBherebyBhenrichB
henceforthB	hemworth,BhempBhelped,Bhelm,Bhelinki-baedBhelbizBheirBheine,BheilaBheetzBheel.BheckBhebaBheavy-handedBheavily,BheavieBhealthy.B	healthierBheadquarter.B	headache.Bhe(256)BhdBhayekBhave?BhauntBhaun,Bhauer,BhauerBhauckBhatyBhatterBhatredBhatner,BhathormmBhater,Bhate.Bhat.BharneingBharmonyBharmon,BharktronBhariah-compliantBharhitaBhared.Bhardware-baedBhardlineB	harbingerBharaingBhappilyB	happened?Bhape,Bhanyecz,BhannityBhanmugaratnamBhangukBhandyBhandout.B	handomelyBhandgunBhandetBhandelblattBhandcah,Bhana,BhamterBhamphire-baedBhammerBhamirBhamBhalvorenBhallmarkBhalimehBhalifaxBhale,BhakyBhakerBhaiti,Bhair,BhadeBhacking.BhackenB
hackathon,Bhack;BhaciendaBhaabotBh.r.BgyftBgxBgwadabeB	gundlach,Bgun,Bgum,Bgujarat,BguidingBguide:BguaidBgtxBgtaBgrown,BgrillBgriderBgrey,B	greenwichBgreenpage.cahBgreenlitBgreeneBgreed,BgrauB	gratkowkiBgrappleBgraph.BgranularBgranBgram.Bgram,BgraderBgracieBgraberBgpBgovernment?Bgovernment-mandatedBgolonoyBgoliathB	goldmoneyBgoipBgoiBgodfreyB	gocrypto.BgobblingBgo?Bgo-aheadBgnoBgmeBgmbh,BgmbhBgmBgluedBgloveBglobitexB
globalcoinBglitcheBgleanBglauberB	gladtein,BgitBgillyBgileBgildedBgilBgiganticBgiftoBgibonBghot,Bgever,Bgerman-baedBgeoBgenre.Bgenre,Bgeneva,BgeneticBgenerouB
generator,B
generativeB
generally.Bgenei.Bgender,BgehrigBgbxBgbp,Bgazprom,Bgazette.BgazetaBgazeBgayB
gathering,BgatedBgate.io,Bgarden.Bgarden,Bgarbage.BgandhiBgamutBgametop,Bgame-changingBgamblerpick.comBgalore.BgallowayBga.BfxtbB
fxbitinvetBfutianBfundraiing.B
functionedBfuion,BfugitiveBfufuBfuelledBfuedBfuangBfuBftB	fruition,BfruitionBfruit.Bfrot,Bfrom?Bfrog,B
frightenedBfried,BfridmanBfrenziedBfreneticB	frenchmanBfrench-peakingB
freh-facedBfreelancer.Bfreebitco.inBfreddieBfredBfray,BfranticallyBfranko,Bfrankly,BfranklinBfranek,B	franchie.Bfranc.Bframe,BfoxminerBfox,Bfourth-highetBfourfoldBfour-dayBfoundry.Bfounded,BfoteredBforward-lookingBforth.Bforret,BforretB
formidableBformed,B
formalizedBforklog.Bforked.Bfork?Bfork.lolB
forgotten.BforgotBforgivenBforgeryBforgerBforex.Bforever,B	foreignerB
forefront.BforeB	football.BfontBfomo3dB	fomentingBfolhaBfoldedBfoldBfnB
fluctuatedBfloatBfliptarter.cah.B	flinttoneBflexaBflaredBflag.BfixeBfix.BfivefoldBfive-tarBfive,Bfitzpatrick.BfittingBfirt-claBfirm:Bfirefox,B
fireblock,Bfirano,BfintracBfinma-approvedBfinma,BfinkBfiniko.Bfine-tuningBfinding:Bfind.Bfind,B
financieroB
finalized.B
finalized,Bfilm.Bfighter,BfiercelyBfideBficorBfiat-backedBfew,BfernandoBfentanylBfengBfence,BfenbuhiBfeminitBfellerBfeityBfeet.BfeeconBfednowB	federallyBfed.BfebBfeature:BfeatherBfdic-inuredBfd7BfaxB	favorablyBfatigueB	farmarketBfarka,B	farewell.B	far-rightBfaqBfanzone,BfanzoneBfantay.Bfanfare,Bfallout.Bfalling,BfalehoodBfake,Bfaith,Bfacinating,BfabricBfabianBfaberBfaatBeymourB
exuberanceBextrudeBextremitBextradition.BextortionitBexteriorB	expreion,B	exportingBexporterBexploitableB
exploding.BexplodeBexplaining:BexperimentedBexperienced.Bexpene,Bexpedition,B
expanding.Bexpand.Bexpand,Bexodu.BexmarketB
exitentialBexited,B	exhautionBexerciedB	exemptingB
exemplifieB	excluded.B	exciting,Bexcitement.BexchangewarBexchange-traded-fundBexamination.BexaggerationBexactly?Bex-wifeBewanu,BewanuBevrofinanceBevolutionaryB	evocativeBevil?BevidentiaryB
evidencingB	eviction,Beverywhere,Bever-evolvingB	even-pageBeven,Beve.Beve,Bevaluation.B	eurozone.BeuroytemBeuropaB	eurogroupBeurexBeugeneBettled,Bettle,Betn.B
etimation,Betimate,Betho,BethiopiaBethi,B	ethermineBethereum-poweredB	ethercan,Bether-dollarB
ether-baedBetf-likeBeternalBeteemedBetaBerikenBerectBerc-721BerbianBeratzBeraeBequity.Bequitie.BequinorBequelBequatedBepouedBeport.B	ephemeralB
eparately,Beofinex,Beo.ioBenzymeBenviouBenviagedB	envelope.BenuredBenuedBentreleaderhip,B	entirety.B	enthuiam,BenthrallingBentertaining,BentertainedB	entertainB
entencing.B
entencing,Bent,BenrolledBenrichedB	enormoulyB	enjoyableBenior,Bengland-baedBengineering.B
engineeredBengenderBengagement,B	enforced,Benergy-efficientBenergieBenemy.BenegixBenegal.Bendorement,Bended.B
endeavoredBend?B
encrypted,BencouragementBencapulatedBenationBemurgo,BemptiedBempowerment,Bemphaizing,B	emphaize.BemotionallyB	emotionalBemojiBemittedBemirate.BeminoleBemiion,Bemeter.B
emeryvilleBemerituB	emerging.Bemerged.Bemerge.B
emboldenedBembodieB
embezzlingBembezzlement.Bembezzlement,BelwoodBellithorpe,B	elliptic.BellipalBell-off,Belki.Belite.B
eliminatedBeliaBeliB
elf-impoedBelendyBelektraB
electivelyBelderBelbaorBelaticBel-erianB
ekuritanceBejectedBeizure,Beion:B
eighteenthB
eight-yearBeiffelBeierBeidooBeibBehrlich,BehooBehaBegyptian-americanBegwit?Begwit.BegoBegalitarianimBegalitarianBefficiency.BeffectiveneBeffectedBeffect:BeetBeential:BeemingBeekerBeek,BeeaBee.BeducatedBedman,BeditingB
edinburgh,Bedge.BedailyB
ecuritizedBecurely,Becuador,Bector:Becrecy.BecortBeconomy?B
economita,BechoingBechoeBechetBecape,Becalate.Bec:Beba,Beay;B	eay-moneyBeattle-baedBeaterieBeatbch,Bearth,BearnetBearner.Bearche.Bearch.BeaonalBeaing.Beaing,Beaily,BeagleBeaedBeacBea,Be-mailB	e-hryvniaB
e-currencyBdytopiaBdynamic.Bdynamic,Bdurov.BdurableBdunworthBduniapayBdungeonBduma.BdukeBduganBduckBdublin.Bdry.BdrowningBdrownedBdropgangBdrop-inBdriver.Bdrivechain,BdreedBdreadedBdramatically,BdrainingB	draftkingBdraft,BdrachmaeBdpwBdpoBdoxxedB	downtime,BdownplayB	download,Bdownize,BdownideB	downfall.Bdoubled,Bdoronin,Bdoor-to-doorBdoor,B	dominicanBdomenicoBdomBdollarization.BdollarizationBdollar-baedBdolevBdolderBdoj,BdogecanBdoge-1BdodgingBdocumented.BdocumentarieBdocketBdobBdncBdnbBdmytroBdmexBdmcc;BdmcaBdk.Bdj,BdizzyingBdizzyBdixon,BdixonBdivorce.Bdivided.Bdivide,B	diverity.Bdive-bombedBdiueBdiuadeB	diturbingBditributed,B	ditortingBditinguihedBditilledBdiruptorBdiputingBdiproportionateBdipped,BdipoedBdipoal.Bdiploma,Bdiplay,BdiplaceBdipenedBdiorder.BdiorderBdiney+Bdimied,BdimenionB
dimantlingBdimantleBdimalBdiligentB
diligence,Bdilemma:BdikB	diipated,BdiintermediationBdiinteretedBdiidentBdignitarie,BdigixBdigitization,BdigitecBdigitalizingB	digitalaxBdigiconomitBdiffer.Bdient.BdieminationBdied:Bdied,Bdie-offB	dictatingB	dicreetlyB
dicreditedBdicount.BdicontinuedBdiconnectingB	diconnectB	dicloure,B	diciplineBdichargeBdiburedBdiamond.B
diallowingBdiallowBdiadvantage.BdevilBdevelop.BdevainiB
detructiveB	detroyed.BdetrimentalB	detrimentB
detokenizeBdetinyB	deterringB	deterrentBdeterredB
detention.B
derivationBderidedBderegulatoryBderegulationBdeprivedB	depreion,BdepreedBdeppBdepot,B	depoitaryBdepoedBdepoeB	depletingBdeplatforming,BdepictedB	departingBdepairBdenverB
denominateBdenied,Bdemontration,Bdemontrate.Bdemographic.BdemocratizedBdemocratizationB	democrat.B
democracy.Bdemocracie,BdelvedBdelta,B
delivered,BdelightBdelgerdalaiBdeire,BdegradedBdegradeB	degenere,Bdefinition,Bdefiant,BdefianceB	defi-baedBdeferredBdefene.BdefeatedBdefactoBdeepakBdeducedBdedic,B
decryptingBdecryingBdecription,BdecriminalizeBdecriedBdecribe:Bdecribe,Bdecree,BdecredBdecreae,B	declined.Bdeclaration,BdeckerBdeciiveBdecided.Bdecided,Bdecent,Bdebt-to-gdpBdebated,B
debatable,BdcmBdcaBdbaBdaymakBdaylightBdavido,BdatukBdarrellBdarknettat.comB
dappradar.B	dappradarBdaniyarB
dangeroulyBdanger.BdancerBdampenedBdampenBdamningBdamienBdamage,BdahjrBdah:BdabbledBd)Bcypru,BcylinderB	cyberpunkBcyberpoliceB
cyberpace.Bcybercriminal,Bcyber-ecurityBcybavoBcxBcvx.financeBcv,BcuyahogaBcutomizeBcutomizationBcutodianhipB	cutodial,BcuteBcurve.B	curtailedBcurryB
currently.B	currency:Bcurrency.com,Bcurrency.comBcurrency-relatedBcuriou.BcurationBcup.Bcult.B
culminatedBcueBcuba.BcryptowhaleB	cryptowarBcryptovere,BcryptovantageBcryptounivereBcryptoquantBcryptophyl.comBcryptophyl,B
cryptonoteBcryptonize.it.Bcryptokittie.BcryptoindutryB
cryptohareBcryptography.BcryptoeconomicB cryptocurrency-to-cryptocurrencyBcryptocurrency-fueledBcryptocompare.comBcryptobuyer,BcryptoanarchyB
cryptoaet,Bcrypto-to-cahBcrypto-tarvedBcrypto-rapperBcrypto-pace.Bcrypto-orientedBcrypto-incomeBcrypto-debitBcrypto-bearBcrypto-avvyBcrypto-anarchyBcryptBcruz,BcruoeBcruhedBcrucibleBcrooverBcronje,BcronieBcrohairBcroatia.Bcriptonoticia,Bcript.BcrippleB
criminallyBcrii-hitB
creenprintBcreen,Bcreek:BcredenceBcraze,BcrawledBcrapedBcrahe.B	craft.cahBcrack.Bcpu.Bcox,Bcowen,Bcovid-19-inducedBcovenantB
courthoue,Bcourt-approvedBcourageBcountrBcountieBcounterpartieBcounterfeitBcounter-economicBcountedB
countdown:BcouncilmemberB
cottonwoodBcott.Bcotland.BcotiBcortezB
correctly.B	corpoelecBcorpionB
coronationBcorentinBcore?BcorcoranBcorairBcopyrightedBcopper.BcopiouBcope,BcopaBcooletBcoolbitxB
convolutedBconviction.Bconviction,B
convicted,B	conveyed:B	converterB
converion,B
converely,Bconvention,Bconvenient.BconveneB
conundrum,B
conummatedBconultation,BcontruedBcontrolled.BcontradictionBcontractor,B	continualBcontext.Bcontext,B	conteoto,Bcontentiou.BcontemplatedBconpiredBconpicuoulyB
conortium.Bconolidate.Bconole.Bconole,BconobBconnectivity,B
connected.B
conidered.BcongratulateB	confuing,Bconfirmation,BconervationitBconervationBcondominiumB
condemningBcondemnB
concurrentBconcurB
concordiumBconciouBconcert,Bcon,B
computing,BcomputerizedBcomputation.BcomputationB	compuloryBcompreBcomplicationBcomplexitieBcompletion.Bcompletely.B	complete:BcomplainantB	compilingBcompetedB
compendiumBcompenatoryBcomparativeBcompany:BcompaionateBcompBcomoroBcommunity-ledB	commotionBcommon?Bcommit,BcommerciallyBcommercializationBcommentary.B
commencingB
commandeerB
combinatorBcolonial-eraB
colocationB	colludingBcolloquiallyBcolliionB
collaping.B	collaped.Bcollab+currency,Bcoinwarz.comB
coinquare,BcoinpotB	coinplug,Bcoinone.BcoinmarketcalBcoinmallB	coinkite,B	coinjoin.BcoinjarB	coinhive,Bcoingeek.comBcoingecko.comBcoinfluxBcoinfairvalue.comBcoinedB
coinecure,BcoindarBcoincidentally,Bcoincidence.BcoincheckupBcoinbit,BcoinbitB	coinbene,BcoinbeeBcoinatmradar.com,Bcoinatmradar.Bcoin:B
coin.danceBcogenerationBcoffeyBcoffer.Bcoffee,Bcoeure,BcoercedBcodBcobainBcoal,B
co-mingledB
co-managerBco-foundingBco-founded.Bco-exitBco-chairmanBcmpBcmoBclydeBclutchBclothe.Bclothe,BclonedBcloggedBclogBcloer,B
clockwork.B	climbing,Bclimb,BclicheBclerBclementeBclearly,Bclearinghoue.B	clearing,BcleaningBcleanedBclaueBclaudioBclara,BclamorBclahedBclaedBcivic,B	city-baedBcitizenhip.B
citigroup.Bcircuit,Bchwab,BchurningB	churchillBchunBchultze-kraftBchulman,BchuchiB	chronicleBchronicBchritodoulouBchritma,B
chritenen,B	chritantoBchoun,BchoreBchoppedBchopBchooe.BcholzBchoiBchino,B	china-runBchile.B	children,Bchicago,BcheterBcherryBcheleaBcheerfulBcheerBcheddar,B	checkout.BchechenBcheatingBchatter.B
chartered,Bchargeback,B	changchunBchang,BchandlerBchancellor,BchairmanhipB
chainfeed,BchainedB	chainbyteBchaiBchBcftBcfd.B	cetinkayaBcetB
certiorariBcertik,Bcertification,Bcertificate,B
certaintieB
certainly,Bcertain.B	ceremony,BcerBcepB	centurie.B	centurie,Bcenored.Bcenored,BcenicBceltic,BcelticBcelo,BcdtBccrfBccidBcbrB
cautioned,Bcaucu,B
categorie,Bcatatrophe.B	cataloniaBcary,BcarvedB
carpoolingB	carolina.BcarnivalBcarnage,BcarltonBcardinalB	cardboardBcarcity,BcarcerBcaptchaBcaproniBcapitulation.BcapitalizedB	capitalimBcapability,Bcap)BcaoBcannot,B	cannazon,BcannabidiolBcannabi.Bcannabi,Bcandy,Bcandle.B
candinaviaBcandalouBcancellationBcampaigningB
cambridge,Bcam:Bcalm,B
calibratedBcaledBcalculator.Bcalculation.Bcalculation,B	calamity.B	calamitieBcake,BcainBcahualBcahportBcahpay.BcahoutBcahier,BcahgameB	cahcript,Bcahback,Bcah2bitcoinBcah-likeBcah-dbBcaeyBcabeiBcaa,BcaBc:goBc2cBbzx,BbytetreeBbytecoinB	byproductBbypaeBby.BbviBbv)B	butterflyB
butterfillBburundiBburntB	burgerwapBburgeBbureaucracy.Bbundle,BbumpyBbumperBbullyingB	bulgaria.Bbulgaria-baedBbukele.BbuinepeopleBbuinemenBbuine-to-buineBbuine-friendlyBbuilt.Bbuilder,Bbuild.BbuhfireBbuhari,Bbug:Bbuffet,BbufferB	budgetaryBbuddyBbudd,BbuddBbudapet,Bbuda.comBbuck.BbucharetBbu.BbtrBbtm,BbtfBbtcwBbtcvBbtcpBbtchortBbtcboxBbtc-udBbtc-poweredBbtc),BbryanBbruteBbrothelB	brooktoneBbrookingBbronzeBbroken.Bbroken,B
broad-baedBbro.BbritowBbritih-foundedBbritain.Bbritain,BbrinBbrigade.Bbrigade,BbridgedBbrexit.BbrewBbreakthrough.B	breakout,BbreakerB	breached.BbravioB	branding,B
brandihingBbramBbraiinBbragBbradleyBbracket,Bboxe,Bbowl.B
bounty.cahBbounty.Bbounty,B	bountifulBbountie.Bbountie,BboroughBborn,B	boringdaoBbordierB	bookmakerBbonu.BboniaB	bond-likeBbonaBboltonB	bollywoodBbolivia,BbogartBbodie,BboatedBboard:BboaoBbluetooth-enabledBbluekyBbloq,BbloombergquintBbloom.Bbloom,B	bloodbathBblogpotBblocktvBblocktationBblocker,BblockchainedBblockchain.com.Bblockchain)BblockbidBblockbattleBbloc.Bbledel,BblakeB
blackrock.B	blackout,BbkcmBbizarroB	biweekly,Bbitwage.BbittradeBbittorrent.B	bittlecatBbitterBbittenBbitquareBbitprim,BbitoaiB	bitnovotiBbitnovo,B	bitintantBbitingBbitinfochartBbithubB	bitfunderBbitfrontB	bitforex,BbitforexBbitfiBbitfarm.BbitexBbitdb,Bbitcointreaurie.org,BbitcoinpythonBbitcoinocracyBbitcoinizationBbitcoininfo.ruBbitcoinfee.cahBbitcoincahjBbitcoincaher.orgBbitcoinbanditBbitcoin;Bbitcoin2B
bitcoin.deBbitcoin.com:Bbitcoin-poweredBbitcoin-futureBbitcoin-fundedBbitcoin-denominatedB	bitcoin).B	bitcartccB	bitcache,BbitbuyBbitbay,Bbitbacker.ioBbitaneBbitaiaBbit.com,Bbiq,Bbip21BbiotechvalleyBbion,B	biographyBbiographicalBbio.Bbinance.comBbillingBbilbaoBbike,BbihkekBbig,BbifurcationBbicconB	bicameralBbibox,Bbia,Bbi-directionalBbhatiaB	bhardwaj,BbhardwajBbgpBbezogiaBbezo,B	beverage,BbetellerBbetchange.ruBbernardBberenzonBbenzingaBbenonBbenoB
bengaluru.BbenetBbenefittingBbeneficiarie.Bbeneficial.Bbelow)B	believer,Bbelief,BbelheB	belatedlyB
belarubankBbeide,Bbehind,BbehetBbehaviouralBbeetBbeer,BbeemerBbeefedBbeatenBbeat,Bbearih.Bbear,BbeanieBbeach.BbdipBbcraBbcnBbchgallery,B
bchgalleryBbchd.Bbch.ggBbch-friendlyBbch),Bbbva,BbbtcBbazookaBbazaar,B	baydakovaBbatteryBbatman,BbationB	batching.B	barwicki,BbartoolB	barterdexBbarrier.Bbarbuda,BbarataBbanky.B	bankrupt,Bbanker?Bbanker:Bband.Bbancoagricola,Bballot,B
ballentineBbalerBbale.BbakingBbairBbailout.BbailinBbailey,Bbail-outBbaic.Bbahama,B	bagholderBbaementBbadaraBbaconBbackup.BbacktoryBbacklog.Bbacking.Bbacked.B
backdooredB	backdoor,Bbaby,BbaaBbaBb2,Bb2Bazerbaijan,BayncBaylumBaylor.Baying;BaxionBaxeBawry.BawkwardBawardingBawaitedBavvietBavoid.Baver,Bavenue,BavanzaBavailability.B
autralian,Bautralia-baedBautopyBautomobile.B
automated,Bauto,B
autin-baedBaurB	aumption:Bauditor,Baudio,BatuteB
attribute.Battractive.BattetedBattetation,B	attainingB	attacked,BatrologyBatomicoBatlB
atifactoryB	atellite,Bat.BarweaveBaruB
article.**BarrearBaroueBarnhem,BarmentaBarmenianBarm.BarkanaBarizona.Bargentinian,BardoinoBarchive,B
arbitrage.B
arbitrage,BaraujoBarafuBarabicBarabia,Bapr.BaprBapproaching.Bappreciating.B
appointingBapply,Bapplication)BappendedBappearance.Bappear.Bapparel,BapparelBapp:BapocalypticBapmex.Bapi3BapacBanwered:Banwered.Bantonopoulo.BantitheticalBantitheiBantifragileBanticipated.Banticipated,Banti-terrorimBanti-tate.comBanti-media,Banti-manipulationB
anti-fraudBanti-competitiveB	antander.Banonymouly.BanointedB	annually,BannieB
annexationBankB
animation,BaniBangolaBangle,BangelaBangel.BaneB	andwichedBandlerBandileB	anchoringB	analyticaBanalogyBanalogieBanalogBamuraiBamun,Bamuel:B	amterdam,BamplingBamparoBamourai,BamoaBamm,Bamld5Bamerican-baedBamelieBame:BambiguouB	ambaador.BamatilBamandaBalzaB
alvadoran,BalumniBaltyBalto-headquarteredBalto,BaltoBaltillyBalt,B	alphabet,BalludedBallottedBallianz,BalleviatingBall-time-high,Ball-tarBalive.Balipay,Balina,BaliciaBalibaba,BaliaB
alexandriaBaldenBalbum.BalbertaBalbaniaBalary.BalariedBalarie,BalamancaBalamBalagoriaBakoin.Baked.B	akari-payBakaneBakahBaitance,BairwapBairtm,BairtimeBaireBairbnb.Bairbnb,Bairbitz,BainineBaignBaifedeanBaide.BaicioBaicboot?Baicboot.Baic,B
ai-poweredBai-baedBagutinBagriculture.BagreeingBagree.Bagorit,Bagora:BagingBaggregator.Bagency:B	afterwardBafpBafootBaffronBaffordable,B
affiliate,B
affidavit.Bafari,BaetheticallyB
aetdah.comBaet:BaeropaceB	aemblymanBaemblingBadviory,Badvertiement,Badopter.Badmittedly,B
admittedlyBadmirerBadminitrator.BadidaBadheringBadherentBadenaBaddledBaddictBadd-onBadaptedBadam,Bad:B	actually,B	actualityBactre,B
activated.Bacronym,B
acrificingBacquire.Bacquire,BacquaintanceBacquaintB	ackerman,BackB	achieved.BachaB	accurate.B	accurate,Baccumulation,Baccounting.Baccount?Baccomplihment.B
accompliheB	accompanyB
accompanieBaccommodation,BaccoladeB
accidentalB	acceorie.BacceorieBaccelerator.Baccelerator,BaccelerationB	acceible,Bacceibility.BacccB	acapulco,BacapulcoBacademy,BacademiaBaburdBabrinaBabreatBabove.Babove-mentionedB	abounded.Babound,B	abolitionBabokifxB	abkhazia,BabigailBabence,B	abdulhaanBabductedBabating.B
abandoned,BabaBab33Baaron,Baange.BaamanBaadBaa,Ba:Ba-zB[will]B[to]B[of]B[itB[forB[cryptocurrencie]B[centralB[but]B90,000B90,B9.9B9-yearB876,000B80.B8,625B7amB7:30B7:00B7:B796B789,000B785B777B75,000B740B73,000B70,B7.95B7.9B7.8B7.76B7.1B7-nanometerB	7-eleven,B7,200B6:00B69,369B69,000B68,B67,000B666B66.B650B65,000B623B611B61,B600thB6000B6.7B6.2B6-7B6-12B5kbB5:30B5:00B58,000B570B560B56,B55.B523,000,000B52,B51tB50xB50kB50cycle,B504B5031B501(c)3B500kB50-dayB50-60B5.9B5.7B5.48B5-yearB5-15B5,200B5,050B5)B4th,B4e2dB497B494,784.B493B49,B483B48-hour,B48,000B478558B475,000B46.4B450,000B45.8B444,000B432B421B42-year-oldB414B411B41.5B40-70B40,B4.1B4-8B4-5,B4,709B4,700B4,400B4,300B4,200B4)B3rd.B3iq,B3commaB381B380B355B354B350,000B34,000B330B33-year-oldB32,B314B310B31.9B30ebB3060B30+B3.3B3-6B3-5B3,654B3,600B3,300B3,085B2fa,B2:14B290B29.5B29,646B28-29,B28,000B277B27-28,B26th,B2658B26-year-oldB25th,B2585B25.7B246B2400wB240,000B24-yearB24-wordB24-7B233,000B232B23-25B23,000B22ndB22:B2265B226-byteB226B221B21e8B2194B213B212B21168B21-year-oldB21,454B20kB	20b91,000B20ac5B	20ac11.5mB20ac100,000.B2029.B2026anB
2026again.B2023B202B201dblockchainB201cytemB201cyourB201cyeterdayB
201cwithinB201cwidepreadB	201cweirdB
201cwalletB201cviolationB
201cvictimB	201cvalueB
201cupportB201cupendedB201cunuuallyB201cunregiteredB201cunpermiionedB201cunlawfulB
201cunfairB201cundermineB201cultimatelyB201cufferedB201cuB	201ctupidB	201ctulipB
201ctryingB
201ctrutleB
201ctrutedB201ctrongetB201ctrengtheningB
201ctravelB201ctranitoryB201ctraditionalB	201ctradeB201ctimeB
201cthreatB201cthoeB	201ctheirB201ctemporarilyB201ctechnologyB201ctaxationB
201ctartupB201crunB201cruleB201crikyB201cregulatorB
201credeemB201credB
201crecordB
201creckleB201creceivedB	201craiedB	201cquiteB	201cquawkB201cquantumB201cputtingB201cpureB201cpumpB
201cpublicB201cproprietaryB
201cprofitB201cproceedB201cprimarilyB201cpreliminaryB201cpoweredB	201cpowerB	201cponziB201cpoloniexB201cpoitiveB201cpermiionleB201cpendingB201cpenaltieB201cpeer-to-peerB201cpartneringB201cpandemicB201cpaidB201cotcB201coriginalB201corderedB201conceB201cometimeB201cofferingB
201cnormalB201cneutralB201cneedB201cnecearyB
201cnearlyB201cnearB201cnaturalB
201cnaive.B	201cnairaB201cmultipleB201cmuchB201cmonthlyB201cmimblewimbleB201cmileadingB	201cmiingB	201cmightB
201cmeaureB
201cmarvelB201cmaliciouB201clong-termB	201clocalB
201cliquidB201climitedB	201clegitB201clegalizationB	201clayerB201claunchingB	201claterB	201clamboB
201ckillerB201cjayB	201cjapanB201cixB201ciuedB201ciueB201cirrationalB201cintroductionB
201cintendB201cintangibleB201cintact.B201cinnovativeB
201ciniderB201cindependentB201cincreainglyB201cincreaeB201cin-depthB201cimplicationB
201cimilarB201cimB201cillB201cignB	201chuobiB201chotB201chortB201chigh-levelB201chackB
201cgoldenB201cgodB201cgenerationalB201cgeneralB201cfuturiticB
201cfutureB201cfundamentalB201cfreedomB
201cfraud.B	201cfoterB201cfollowingB	201cflokiB201cflahB201cfirt-everB201cfindB201cfedB201cfeatureB201cfarB201cfaleB201cfairB201cfailureB201cfacebookB
201cexpandB201cexitingB201cexitB201cevidenceB201ceverydayB201ceverelyB201ceventuallyB
201ceuropeB201cethB201cerieB201cepicB201cenvironmentalB201cenhanceB201cencourageB201cempowerB201celf-regulationB201celectricityB201ceemB201ceducationalB201ceatB
201cdollarB201cdiruptiveB201cdidB201cdicuB201cdevelopmentB201cdedicatedB201cdarkB201cdangerouB201ccryptocurrency.B201ccroroadB	201ccrimeB
201ccreditB201ccrackdownB201ccoreB201ccontinuingB201ccontinueB201cconfirmationB201ccollectiveB	201ccloerB	201ccleanB
201ccircleB	201cchiefB201cchargedB201ccertainB201ccbdcB201ccapitalB
201ccanadaB
201cbuyingB	201cbuineB201cbuildingB
201cbreezeB201cbootB201cblockchain-baedB201cbitfinexB201cbetweenB201cbelieverB201cbankingB201cbackingB201caxieB201cauthorizedB201cauthorityB
201canyoneB201canatomyB201calthoughB201calternativeB201caloB201caiaB201caggreivelyB201caggreiveB201caetB201cadvancedB201cadvanceB
201caddingB201cachieveB201caccidentallyB201cacceB201cabueB201c[bitcoin]B201c2018B2019writingB2019weB	2019etichB2019ed,B201921B2018xrpB2018workingB2018whyB	2018whiteB2018weB2018undoB
2018trutleB2018tranactionB	2018torchB	2018tipprB2018repreentativeB2018projectB	2018ponziB
2018pirateB2018paymentB2018patternB2018optimizationB2018oldB2018noB	2018moneyB
2018miningB2018mcB2018lightningB2018leadingB
2018largetB2018invetmentB	2018indiaB	2018indexB2018hodlingB2018heroB2018hahB	2018greatB2018goodB
2018finneyB2018feeB2018faketohiB2018explodingB2018ethererumB	2018econdB2018dragonmintB2018digitalB2018coindaddyB2018chritmaB	2018chainB2018bullB
2018bottomB2018bobB
2018bitconB2018behind-the-meterB2018bchB	2018aheadB2018aB201851B2018)B2017:B	2017-2018B2016:B2015/849B	2014whichB2014thailandB	2014atohiB201B200xB200fB200eB200atheB200aintercontinentalB200a,B2007,B2006,B2006B2005,B2005B2002.B20/20B2.5.B2.14B2.1B2.0:B2.0)B2-yearB
2-out-of-3B2-6B2-5B2,200B2,100B1mb,B1:30B1998.B1998B1997,B1995B1994B1993.B1992B1991.B1990.B1989,B1989B1987.B1984,B1981B1980,B1970.B197B195B1944,B1934,B1900+B19.6B19-year-oldB18:15:05B187,000B186B185B180,000B18.6B18.3B18-yearB17:B174B1709.B17.8B17.3B17.2B17-year-oldB17-monthB16nmB166B162B16-17B15xB15:B158B156B155B152B1500B15.5B15-20B14th.B148B144.B142B1400B14.8B14-dayB132B131B12:B128mb.B125,000B1241B124B122B12.6B12-year-oldB12-hourB12-24B
12,965,000B11xB11:59B11:00B116B114,000B11.8B11.5B10th,B10amB10:00B105B1000x.groupB10-monthB	10-minuteB10-kB10-foldB10-dayB10-15B10+B1.8.B1.25B1.0.B1-to-1B1-3B1-15B1,700B1,650B1,281B1,179B1,000,B0bbeB0bb5B0bb1B0bafB0ba4B06ccB064aB0648B0628B0501B0443B0442B043bB043aB0439B0422heB0415B010dB00fdB00fclB00f6ereB00f3n,B00f3mezB00eetB00edvar,B00edaz-canelB00eadoB	00e9chet.B00e7aoB00c9B00a7B00a3500B00a0zimbabweB00a0zhaoB
00a0zebpayB00a0yearB00a0xbtB00a0writtenB
00a0withinB00a0whoB	00a0whileB	00a0whereB00a0whenB00a0wallB00a0wahingtonB
00a0volumeB00a0uingB00a0uedB	00a0tudy:B
00a0traderB00a0teveB00a0tayB
00a0tartupB	00a0riingB00a0reward-baedB
00a0reviewB
00a0reviedB
00a0reuterB00a0report,B00a0remittance-demandingB00a0remittanceB
00a0reerveB
00a0redditB00a0realB00a0rbiB00a0quetionB00a0qeB00a0profeorB00a0previoulyB00a0planB	00a0peterB	00a0panihB00a0paidB00a0notB00a0nigerianB	00a0niallB00a0myB00a0mtB00a0motB00a0mining,B	00a0miamiB	00a0mediaB00a0markB	00a0maltaB00a0malayianB
00a0makingB00a0maintreamB00a0macranomB00a0maachuettB00a0lotB	00a0localB00a0litecoinB00a0lawB00a0italianB00a0irinB00a0invetorB00a0interetB00a0initialB00a0ingapore-baedB00a0indopaceB00a0illicitB
00a0highlyB00a0helpB00a0hardB00a0gwynethB00a0growingB	00a0greekB00a0governmentalB00a0gibraltarB00a0germanyB	00a0geneiB	00a0fear,B00a0examiningB00a0evenB
00a0etoniaB00a0endB00a0embracingB00a0electronB	00a0egyptB	00a0earchB
00a0duringB
00a0devereB00a0deutcheB	00a0deathB00a0dancingB	00a0dailyB00a0cryptobuyerB00a0criticalB00a0contentB00a0conideringB00a0companieB00a0commodityB00a0colombiaB00a0codeB00a0claB00a0chicagoB
00a0cex.ioB00a0ceoB	00a0buineB00a0bloombergB00a0bitpointB
00a0bitmexB
00a0bitkanB00a0bitgrailB00a0bitcoin:B00a0bitB
00a0bankerB00a0articleB00a0antpoolB
00a0andreaB00a0analyt:B	00a0amungB00a0altcoinB	00a0allowB00a0allB00a0aimB
00a080,000B00a05B00a03B00a0200,000B00a020+B	00a012.6mB00a01B0.7B0.60B0.4B0.35B0.25.B0.25B0.19.0B0.18.0B0.18B0.1.B
/r/bitcoinB-1xB-19B-0.5B**updateB**theB(zec).B(yfi)B(xvg),B(xvg)B(withB(whoB(wh)B(vfa)B(utc),B(unicef)B(udh)B(ub)B(uae)B(uB(tyo:B(txv:B(twe:B(tvl).B(try)B(trx),B(trp)B(tp)B(to),B(tdlc)B(tbb)B	(tavanir)B(tae)B(rub)B(rti)B(rpg)B(r).B(qe).B(ptr),B(profeorB(pcc)B(pc)B(pax),B(onlyB(ol),B(oft)B(nye:q),B(nyd)B(nrB(nl)B(ngo)B(nfa)B(nem)B(nearlyB(ndrc)B(nbu),B(nav)B(nadaq:mara)B(naaa)B(mmo)B(mit).B(mena)B(matic)B(marketB(luna),B(kwh).B(kra)B(knownB(j/th)B(ipf).B(iot),B
(includingB(iif)B(ieo)B(ict)B(i.e.B(iB(hrf)B(hockingB(gud),B(gpu).B(gdpr)B(gdp)B(gbx)B(fx)B(fud).B(ftc)B(formB(finra).B(finma),B	(fincen),B(fil)B(fig.B(fee)B(fed)B(fdic),B(fda).B(fda)B(fb).B(evm)B(etn),B
(epeciallyB(en)B(emn)B(ema),B(egwit),B(eea)B(eda)B(ecb).B(eba).B(dx)B(dpo)B(dot),B(do)B(dnb)B(dk).B(dk)B(dfa)B(dbaB(dai),B(dah).B(daa),B(ctor),B(cryptocurrency)B
(covid-19)B(conob),B	(commonlyB(cnmv)B(cmp).B(cmc),B(ceoB(ce:B(cdv),B(cd),B(cd)B(cctv-2)B(ccb)B(cbk)B(cba),B(cac)B(ca)B(c),B(bv).B(buyB(bti)B(btcp),B(btcB(bp).B(bornB(boj),B(bofa)B(blackB(bit),B(beijingB(bcra)B(bchn)B(bchB(bcc).B(bcc),B(bca)B(bc),B(baa)B(atohiB(app)B(amd)B(aic).B(ai),B(afp)B(adgm)B(accc)B(a16z)B(7),B(24),B(2).B($101B($1.1B$yfiB$pntB$edoB$btcB$97B$96B$950B$923B$90,000B$9,940B$8.7B$8.3B$8,600B$7bnB$78B$778B$776B$74B$7200B$7.4B$7.3B$7,365B$6mB$696B$670B$67,017B$64,804B$635B$611B$61,782B$602B$600kB$60,000B$5mB$5bB$5800B$56B$540B$52kB$52B$515B$51,000B$50k.B$5000B$500,B$50,000.B$5,700B$5,118,266,427.50B$4bB$48,000B$47kB$470B$46kB$450B$44,846B$43,000B$42kB$420B$4000B$4.9B$4.5kB$4,000,B$38kB$36kB$336B$330.6B$33,000B$32.17B$31kB$300kB$3.25B$3.19B$2milB$290B$286B$28,800B$28,000B$2760B$272B$27.6B$266B$26,000B$25kB$255B$250mB$250kB$2500B$24kB$246B$23kB$226B$223B$221B$215B$209B$204B$202B$200.B	$200,000.B$2.33B$2.21B$2,800B$186B$180,000B$17kB$177B$172B$171B$17,000.B$17,000B$16bB$168B$160,000B$16.8B$15mB$1567B$154B$1500B$150,B$15,000,B$147B$145mB$144B$141B$1400B$14,900B$13bnB$138B$13.7B$13.6B$13,900B$13,000.B$129B$126B$121B$120,000B$12.7B$12,000.B$114B$113B$112B$11.6B$11.5B$11,900B$11,600B$104B$10.0B$10,400B$1.76B$1.55B$1.47B$1.3bB$1.311B$1.21B$1.19B$1.07B$1,500,B$1,236B$0.731B$0.12B$0.06B$0.049B$0.02B$0B#dogecointothemoonB#congrepaubiB~B|BzytaraBzykind,BzybiBzwetBzubrBztorc.BzrxBzookoBzoo,BzodiacBzobelBzk-nark.Bzk-nark,BzivBzippingBzipBzionBzimwitchBzimbabwean,B	zimbabwe:BzigzagBzhongjiaBzhengBzhavoronkov,Bzhang,Bzhan.Bzeronet,BzeronetBzengoBzellBzelaya,B	zeitgeit.B	zeitgeit,BzeitgeitBzealouBzealotryBzealotBzealand.Bzealand-baedBzeB	zczepanikBzcfBzarooniBzardBzapitBzanzibarBzambia,BzambiaBzainabBzaBz.ByuryByuliaByuko,ByuhByuguda,ByugudaBytemair,Byria.B	youtuber.B
youtube-dlByouth.B	your.org.Byou!Byonhap:B	yong-jin.Byokoo,ByokoByohiyukiByobit,B	ynthetix,B	ynthetic.B	ynonymou,BynergyBymbolimBymbioiByloByingkouByifeiByield-farmingByfi.ByevgenyByemenByekta,ByehByeecallByeatByear;Byear-over-year.Byear-over-year,Byear-on-year.Byear-on-year,Byear!Bye.Byap,ByapByaoByangtzeByandex.Byale,Byakuza.Byacht.BxzavyerBxuedong,BxtremeB	xthinner,BxrpiiBxrp/ud.Bxpt.BxportBxpay,BxoelBxmr.BxmBxlm.Bxinhua,Bxigma,B
xiaochuan,BxiaoBxiangminBxhbBxfinityBxetra,BxerBxelerBxdaoBx11BwyborczaBwwfBwuytBwuille.Bwuille,BwtiBwronglyB
wrongfullyBwrongfulB	wrongdoerBwritten.B	writing).B	writedownBwreckBwrangleBwpnBwpcryptoexchange.comBwpcryptoexchangeBwound.Bwould.BwotokenBworthwhile.Bworth?Bwormhole.cahB	wormhole,BwormB
worldwide?B
worldview.B	worldlineB	worldcoreBworld;BworkpaceBworkingforbitcoin.comB
workforce,Bwork-relatedBworhipBwordpre.Bwoolard,B	woodwork.BwoodwardBwont.Bwont,B
wondering.B
wondering,Bwoman.BwolverhamptonBwolfgangBwokenBwojcickiBwobblyBwizardBwivBwittrockBwithholdingBwithheldBwitherBwithdrawing.B	withdraw.BwitfullyBwitch.B	wirtchaftBwirlingBwired,Bwiquote.Bwiped.Bwine,BwindlingBwind,BwimmingBwimBwillitonBwillieBwill:BwikitribuneB
wikipedia.B	wikimediaB	wikileak.B	wikileak,BwihfulBwihe,Bwih.Bwift.BwierBwidom.Bwidom,Bwidget,Bwicoin.BwhodBwhmBwhitleblower.Bwhitle-blowerBwhitleB	whitelit,B	whitebit.B
whitebird,BwhiperedBwhim.Bwhim,BwhikeyBwhereabout.BwheatB
whatminer,BwharfBwhackdBwhBwgcBwetwardBwetpac,Bweth,BwethBwertpapierhandelbankB
wertheimerBwernick,Bwere,Bwenheng,B
wenatchee,Bwelling.Bwelling,B
wellbeing.Bwell-reearchedBwell-known,Bwell-documentB	welcomed,BweirderBweir,Bweight.Bweight,BwegBwefBweek?Bweek;BweechatBweeBwedgeBwedbuhBwebjetBweb:BweaverBwearyBweapon.Bwealth-xBweakne.Bweaker.BweakenedBweak,Bwe.newBwbdapp.com.BwayedB	wayamevakBwayam,Bway;BwavecretB	watt-hourBwaton.BwatfordBwaterwayB	waterhoueBwater-kiingBwate.BwatanabeBwarringB	warrantleBwarrant,Bwarner,Bwarn.BwarmthBwarming.BwarholBware,Bwaraw.Bwar?Bwar-tornBwapping.BwannabeBwang.Bwaned,BwandererBwandaBwanacryptorBwalterBwalmart.B	wallpaperBwallex.Bwalletconnect.B
wallet.datBwall;Bwall.Bwalk,Bwake.Bwakanda,BwaivedBwaiveBwaiting,BwailBwaifuB
wahtradingBwahoeBwahedBwagon.Bwagging.Bwag.BwaferBwaenteiner,BwadedBwaabi.B	vyachelavBvyBvulcanBvremyaBvr,BvoyageBvoxelBvouchingBvouchBvoting.B	votepeer.Bvorick,BvoremBvoorhee.Bvolunteer-baedBvoluntaryim-baedB	voluminouBvolume;Bvoltaire.cahBvolkbankBvolcano-poweredBvohra,Bvogue.Bvoell,BvnexpreBvneheconombank,BvneheconombankBvladivotok,BvladilavB	vkontakteBvividBviualization,B	vitriolicBvitriolBvital.BvirungaBvirue.Bvirue,BvirtcoinBvirgileBvirallyBviralityBviolent.BvinnytiaBvineBvilniu,B
villaverdeBvillanueva,B	villalba,Bvillage:BviktorBviiting.Bviited,B
viionarie,B	viionarieBvigor.Bvigna,B
vigilante,Bviewer.B	videogameB	victorianBvickerBvice,B	vibrationBvia:Bvi-BviBvflectBveve.BvetoedB
veterinaryB
verticallyB	vertcoin,BveronicaBvernon,BvernonB	vermorel,Bverizon,Bverion-rollingBverifieB
veriblock.Bverde.Bveratility.Bventura.BventingB
ventilatorBventilationBvenrockB
vengeance.Bvending,BvelvelBveltycoB	velocity,BvelaBvekuBvein,B
vedyakhin,BvectraBvebBve,B
vcurrencieBvctrade.BvcaBvbBvaultyBvattenfall.Bvary,BvarnaBvariety,B
variation.B	variance.BvarianceB	variable,B
vaporware.B
vaporware,BvapeBvannaB
vanity.cahBvanity,Bvanihed.Bvanihed,BvaniheBvanel,BvandaBvampireBvaluepenguinBvalue)BvalleriuB
validated.BvaleriyaBvalenz,B
valentine,BvalentinBvaldiBvalarBvalBvailievBvachiraBvaccine,BvacateBvacancyBvaB
v0.1-alphaBv-hapedB
uzbekitaniBuyghurButu,ButuButtarButreexo.ButopianButilieButermannButdButainingBurveyed.Buruguay,B	urrogate,B	urpriing.Burprie:Burpaed.B	urowieckiBurnamedBurmieBurl,BurgeonBurged,BurferB	urfacing,Bure:Bure.Bure-winBur.Bupwing,BupudBuptreamBuptime.BuptartBuppreedBuppreB
upport.comBupplie.BupplantBuppingBupon,Bupload,B	upicioulyBupicion,BuphotB	upholdingBupet.B	uperviingB
upertankerBuperrichB	uperrare.Buperintendent,B	upergroupB	uperfarm,BuperedeB	uper-richBupenion.Bupended,BupectingBupect,Bup?B	unwriter.B
unveiling,B	unveiled.Bunuual.Bunuual,B
unurpriingBununuB	unuitableB
unucceful.BunubtantiatedBuntrue.Buntraceable.Buntraceable,BunthinkableB
untetheredBuntetedB
untenable.Bunret,Bunregulated,Bunregitered,BunrecognizedBunreaonable.Bunrealitic.Bunprofitable,Bunprepared.BunpredictabilityBunpendable.B
unpecifiedBunon),BunofficiallyBunodax.BunodaxB
unnoticed,BunneededBunnecearilyB	unnaturalB	unmatchedB	unlikely.B	unlikely,Bunlawfully.BunknowinglyB	unknowingBunknowable,BunkindB
univeritatBunited.Bunit).B
uninformedBunimportant,BunilaterallyB
uniformityBunicef.B
unicameralBunica,Bunhindered.BunhedgedB
unhackableBunfrozenBunfriendline,B	unfinihedBunfilledBunexpected.BunevenB	unettlingBunequivocallyBunemployed,B	unelectedBuneemlyBunecuredBuneatB
unearthingBundueBundin,BundeterminedBunderwriter.Bunderwhelming.B
undertood,Bundertandably,Bundertaking,BunderreportedBunderpinnedBunderperformedBunderlying.Bunderground,BundergoeBunderdevelopedBunder,B
undefeatedB
undeclaredBunde,B
uncovered.BuncorkBuncontrolledBuncontrainedBunboxingBunbackedBunattractiveB	unatifiedB
unanwered.BunannouncedBunanimouly,BunanimouBunacceptableB	unabated.BunabahedB
ummergirl,Bumar,Bultra-paranoid.Bultra-paranoid,BuleimanBukhoiBuite.Buite,B
uitabilityBuing.Buing,BuimonBuif,BuianiBuhilBuggetiveBugBuffocateBuffer.Buex.BuerbaeBuer:Buer-friendly.Buer-friendly,Buefulne.Bueful.Budh.Buddenly,B	ud-backedBuctBucharitakul,B	uccinctlyBucceor,BucceionB	ucceeded,Bucceed,BubytemBubway,BubtleB	ubtitute,B	ubtantiveBubtantially.B
ubtantial,Bubtance,B	ubreddit,Bubpoena,BubparBubmiion.BublimeB
ubjective.Bubject:B
ubiquitou.B
ubiquitou,Bubioft,BubidizeB
ubidiarie.B
ubidiarie,Bubided.BubideBubi:BubereatBubdomainB	ubcribingB
ub-channelBub-aharaBub-25BuacBua.Bu.k.-regiteredBu..c.Bu..-iuedBu..-compliantBu-zynBu-regulatedBu$5,000Bu$300bnBu$100Btzero,BtzeroB
tyrannicalBtypingBtyon,B	tylometryBtx/day)Btx.BtwofoldBtwo-year-oldBtwo-prongedB
two-optionBtwittertorm,BtwitterphereBtwitter-likeB	twitter).BtwilightBtwenty-threeB
twenty-oneBtwenty-fourBtwenty-evenB	twentiethBtwentieBtwelve-monthB
tweettorm,BtweakBtvl,B	turtledexBturnkeyBturned.BturingBturcoin,B	tupidity.BtuntedBtunt,Btunnel.BtunedBtumpB
tumblebit:B
tumblebit.Btumble.BtuleBtuffingBtuffedBtuff.Btucker.Btubhub.B	tubbornneBttpBttdx,BtrutwapB	trutologyBtruth:BtrutarB
truggling.Btruggle.Btrue?B
tructuringBtruck.B	troutner,BtropicalBtrophy,BtronguBtrolled,BtrolledBtroll?BtroikaBtroiaBtriumphantlyBtriumphBtritanBtriple-1BtrinketB
trillion).Btrike.BtrijoB
triggered.BtrifectaBtrifeBtrickle.BtrickleBtrick-or-treatB
tributariaBtribune.Btribeo,B
tribaliticB	triangle.B	triallingBtriadBtrew,BtretchBtrength.BtrendedBtrend?Btrenche,B	treaurer:BtreauredBtreaty.BtreatyBtreated,Btream:BtreadingB
tre-tetingBtrayedBtraxyBtrawlingBtrawlB	traveler.Btravala.com.Btravala,BtraumaBtratophere.B	trategy&,B	trategit.Btrapani,Btrap.Btranverely,BtranquilBtranportingBtranpoedBtranparentlyBtranliteratingBtranitorBtranitBtranientB
trangreionBtrangleholdBtrangetBtranger.BtrangelyBtrange.Btranformation.Btranformation-relatedB	tranfergoBtrandedB
trancript,B
trancribedBtranaction?Btran-feeBtrait.B	training.BtrailerBtraightforward.B
traight-upBtraight,BtrahnyBtrah,BtragglerB
traffickerB	trafalgarBtraetenBtradingview.Btrading:B	tradewindBtradergroup,Btrade-weightedB	tracking.B	tracking,Btracked,Btracing.Btraced.B	traceableBtpunkBtp.Btp,BtoyingBtoy,BtoxBtownley,Btower,BtoutB
touchcreenBtouch,Btotal-value-lockedBtortuouBtorreyBtorrent.B	torrealbaBtorre,Btoronto.Btoronto-dominionBtoronky,BtormhopBtorjB	torefrontBtored.Btored,Btore-of-valueBtor.B
topy-turvyBtopped,BtopmotBtop-tenB	top-levelB
top-ellingBtop-brandedBtoothBtook.BtonneBtonewallingB	tonawandaBtomeBtomachBtolerateBtoledo.BtolaniBtokuBtokoB
tokenwave,B	tokenlon.Btokenization,Btokenit,BtokeniationBtokenhuffle,B	tokendataBtokenanalytBtoken?Btoken-holderBtokay.BtokayB
tojilkovicBtoilingBtodghillB
tockpilingBtockholder.B
tockbrokerBtock-pickingBtock-for-tockBtochaticBtobolkBtobam,BtoatingBto?BtmxBtmc,Btla),BtizzyBtitularBtitoBtitan,Btireome.BtireleBtippr.Btipping.Btipping,Btipbitcoin.cah.Btipb.chBtintB	tinkerer,BtinkererB
tinkerbellBtingingBtinaB
timulationB	timmermanBtimmer,Btimetamped.B
timetampedB
timepiece,B
timeframe,BtimecoinBtime-conumingBtime!BtiltonBtilmanBtiktok.Btigray,BtigrayBtiglitz,BtiffenBtifelBtier-oneBtidebitBtick.Btice,Btibanne.B	tiananmenBti.Bti,BthrutBthroughput,Bthrough,B
throttlingBthrone,B
three-partB
three-foldBthree,Bthouand.BthorBthodex,BthodexBthirty-threeBthinnerBthinlyBthinker,Bthing;Bthing)Bthin,Bthieve.Bthiel.BthiefBtheymo.Bthereof,BthepdetBtheorem,Bthen:B	themelve?BthematicBthem;Btheblockcrypto.comB
thealonikiBthawBthankgivingBthakur,Bth/.Bth/,Bth/)B	teynberg.Btexting.BtewartBtetraBtetleyB	tetifyingB	tetheringBtether)B	tetflightBteter.Bteter,BtetagroaBtertiaryBterror.Bterritorie,BterrenceBterrainBterpin,B
terminatorBtermedBterm?Bterling.Bterling,BterlingBterahah.Btequila,BtepicoB	tephenon,BtephanieBtenx,Btenure.BtentacleBtent:BtengBtenev,B	tenerife.B	tenerife,BtenerifeBtender:B
tendentiouBtencent,B
temptationBtemptB
temporary.Btemporarily.BtempoB	template.B	tempered,BtemperBteloBtell.Btell,BtelevendB
telephonicB
telephone.BteleconferenceBtelecommunication,BtekceBtejpaul,Btehran-headquarteredBtehran,BtegoeedBtefanoB	teenager.Bteem.Bteem,BteekaBteeBtectonicBtechnological,BtechnocraticBtechnionBtechnicalitieB	technica,Bteam3dBtealing.BtealetBtealer,BteaeBteadfatBteacher,BtcapBtc-baedBtbtc,Btbilii.BtbiliiBtaylor,BtaxonomyBtaxman.Btaxman,Btaxi,BtaxbitBtaxable,Btax-efficientBtax),Btax)BtatyanaBtatitician,BtationedBtatim,BtaticBtatewideBtatefulBtate-owned,Btate-by-tateBtate-anctionedBtartmeupBtarterB	tarkware,Btarkov,BtaringBtariff,BtarcollBtarbuck.BtarbertBtaproot.BtaperedBtapeiro,BtapeiroBtape,BtapcottBtap,Btaotao.BtaoBtantalizing,B
tanrikulu,BtankingBtangibilityBtangB
tandpoint,B
tandardizeBtand.B
tand-aloneBtanberryBtanBtamperyBtamper-reitantBtamper-evidentBtamperBtampedeBtamilBtamford,BtamBtalwartBtallyBtallion,BtallieB	talkativeBtale.Btale,BtalBtakopu.BtakopuB	takeover,BtakehiBtakeaway.com.Btake-offB	tak-forceBtaiwan-baedBtaipei,Btai,BtaiBtahingBtaheB	tagnationB
tagnating.BtagingBtaggingB
taggering,Btage:Btag,BtaennlerBtadgeBtacyBtacticalB
tackmahingBtacklingBtackle.BtacitBtaccBtabloidBtablet,B
tablecoin:Btable-pecificB	tabilize.Btaaki.Bt9,Bt2Bt19Bt17eBt1.B	t-mobile.B	t-mobile,BrzdBryan,Brwanda.BrutyBrut.Brupkey,BrupiahB	runner-upBrungBrunet,BruneBrun-up.Brumaihi,Bruled.Brule:BrukyBrukinBruiningBruinBruian.BruggedBruffer,B	ruenvadeeBruell,Bruck,BrubygemBrubyBrubric,BrubricBrubio,BrubbingBrtiBrt-nftBrt,Brozak,B	royaltie.B	royaltie,Broy.BrowlingBroweBroux,Broute,Bround-the-clockB	roulette.Broubini.BrothbardBroter.BrotbartBrotatingBrotand,BroptenBropeBroot.BroofingBronaldoBromneyB	romanian,BromanBroller-coaterBrokBroiland,BrohanBrohamBrogue.Broger.Brogan.BroenthalB	roenberg,Brodent,Brochard,Brobut,Brobotic,Brobot.Brobomarket,Brobert,Brobbin,Brobbery.Brobbery,BroaredBroadhow.Broad:Bro,BrnBrmitBrk.Brk-ethereumBrk-baedBriyadh,Briver,BrivenBritualBrippedBripio.Brip-roaringBripBrinivaBriky.B	rik-avereBriher,BrihaadBrigidlyBright;Bright-wing,Bright!BriggedBriffB
ridiculou,B
ridiculed.B
ricochetedBricher.Bricher,Briche.Brichard,Brich:B	ricardianBribeiro,BribbonBrial.Brhyme.Brhyme,B	rhetoric.BrgbBrgaxBrezaBreynold:BreynoldBrevolutionizedBrevolutionary,Brevolut.Brevoked.Brevoke.cah,Brevival,B	reviitingBreviion.B
reviewing.BreviewerBrevied.BrevertedBrevered.Breveral,Brevenue-haringBrevenue-generatingBrevelation,B
revealing,BreutzelBreuter:BreurrectionBreurrectBreurgingB
reurfacingB	reurfacedBreume.BreumBreult:BreueB	reucitateBreuchelBretweet,B
returning.Breturn?Breturn-on-invetmentBrettig.Brettig,BretropectiveBretroactiveBretroB	retrievedB	retrievalB
retricted.BretrainBretractBretracement.Bretracement,BretoldBretireB	retentionBretail.B	required,Brepublic-baedBrepreentation.Brepreentation,BrepotedB	reporter:Breportedly,B	reponiblyB
reponible.Breponibilitie.Breponibilitie,BreponderB
repondent.B	reponded.Brepond.Brepoitorie.Breplied.BreplieBreplicatingBreplicaBreplacement.Breplace.B
repetitiveBrepercuion.B	repected,BrepectabilityBrepealedB
reourcefulBreource-richBreorganizingBreorganization,B	reopened.BreonatedBreolved,Breolve.Breno,Brenewal.B	rendered.Bremoval.Bremoval,Bremote.Bremote-firtBremnantBremixpoint,B	remittingBremitanoBremit,B	reminder,BrememberingBremedy,BremediedB
remarking:B	remarkingB	remarked,Bremarkable,B
relocationBrelivingBrelitB	religion.Brelief:BreliablyB	reliable.Breliability,B	relevanceBrelayingBrelaunchingB
relaunchedB	relation,B	relatableBrekt.Brekt,BrejoicedBrejalaBreit.BreinvigoratedB
reinventedB
reinuranceB	reinhart,Brein.BreimpoedB
reimburingB
reimbured.B	reimagineB	reilient.B	reiliencyBreigned.Breigned,B	reidency.Breide.Bregulatory-compliantB
regulator?Bregulation;Bregulation)B
regulated?Bregulate/legalizeBregoB	regitrar,BreginaldBreg.BrefuteBrefund.Brefund,Brefugee,Brefue.B	refrainedB	refocuingBreflection.B
refinerie,BrefinedBrefinancingBrefillBreettingBreetablihedBreet,BreentB
reemblanceBreellingBreeller.BreellerBreellB
reelectionB
reearcher:Breearch-focuedBredwoodB
redundant.BredreBreditributionBredicoveredB	redfearn,Bredemption.Bredemption,B
redeigned.B
redeigned,B	redditor,Breddit:B
red-handedB	recurrentBrecue.Brecruitment,BrecriminationBrecreationallyB
recovered.Brecover.BrecoupedB
recording.B	recorded,Brecord-keepingBreconnectedB	reconnectBreconideringBreconciliation,B
recommend,Brecognition.Brecognition,B	recogniedBreckle.BrechnerB
reception.Brecently-launchedBrecently-announcedBreceive.B
receivableBreceeBreceB
recapturedB	recaptureBrebukeBrebranding,Brebrand.Brebrand,B
rebounded,B
rebittanceBrebitonBrebaeBreapply,B	reaoning.B	reaoning,B
reaonable.Breaon:BrealtyBrealtor,BrealnoeBreally,B	realized.Breality:BrealiticBreaeedBreactorBreactivated.Breach,B
re-openingBre-filedBre-evaluateB
re-electedBrdp,Brbc.B	ravikant.B	ravencoinBraveBraunchyBrattledB	rationingB	rational.BratifiedBrathBrat.Brat,BrarityBraretBrarepepeBrapper.BrappedBrapid7BrantalaBrandom,Brand)Branch,BranchBramonaBramoBramirez,BramiBramblerBramadan,Bram.Brakuten,BrakehBrajivBrajehBrajeevBrain,Braied;B	raidforumBrahulBrahtriyaBrage.BraftaarBraftBraffle.BradkeBradical:BradeonBrada.Bracer.Br3.Br3,Br/walltreetbet,Br/cryptocurrency,B
r/bitcoin.Br&d,Br&bBquote.B
quotation.BquoraBquoinex,Bquo,BquizzedBquizzeBquit.BquirrelBquire,BquintillionBquintetBquinteentialBquinn.BquieterBquidax,BquicketBqueuingBquet.Bquet,BqueezingBqueezedBquebec.B
quarterly,Bquarle,B
quare-footBquantifyingBquantconnectBquandaryBquality.Bqualification,BquaintBquadrillionB	quadraticBquad.Bquad,Bquabble,BquabbleBquaBqqBqinghai,BqinBqiang,Bqfpay,Bqeth11BqcBqatar,BqatarBq.BpywareBpylkkBpyingB
pychology,Bpwc.Bpwc,BpuzzledBputh,Bput,Bpurview.BpurgedBpurge.Bpure,B	purchaed.BpuppyBpupilBpupB
punihment,BpunihingBpunihed,BpuniheBpunihBpundixBpunchedB	pullback.Bpule.Bpule,BpularBpulaBpuherBpuhdataBpuh-onlyBpuffB
publihing.B	publiher,Bpublicly-quotedBpublicly-offeredB	publicizeBpublic/privateBpublic-privateBpu,Bptr,BproweB	provokingBprovocateurB	provider)Bprovide:B
proverbialBproudbitcoinerB	prototarrB
protonmailB	protocol;B	protocol:B	proto.comBprotitution,B
protitute,B	proteter,B	protetantB
protected,Bprotect,B	propulionB
proprietorBproportionalBpropoed,Bpropoe.Bpropoal?BprophecyB
properity,B	propered,B	propectu.Bpropect.Bpropagation.Bpropagation,B	propagateBproof-of-work,Bproof-of-proofB	pronounceB
prompt.cahBpromioryBprominently,Bprominence,B	promiing,Bpromied,Bpromie:Bproliferated.Bproliferate.Bproliferate,Bprojection.Bproject?Bproject;BprogreivelyB	progreingB	progreed.BprogreedBprogramming.Bprogrammer)B
programme.B
programme,Bprofit-maximizingBprofeor.B	profeion.Bprof.Bproductive,Bproduction-readyBproduct?B	produced.BprodBprocuredBproblematic,BprobitBprobedBprobableBproactivelyB	pro-trumpBpro-blockchainB
privilegedB
privatizedB
privately,Bprivate-onlyBprivacy?Bprivacy-mindedB	priority,BprioritizedBprintoutB	printing.Bprinter.Bprinter,BpringingBprincipalityBprimuBprimelyBprimakovBprim.BpriharBpridnetrovianBpriceyBpricewaterhoueBprice?Bprice;B
price-wie,Bprice-fixingB	previewedBprevent,B	prevailedBprevail,BprevailB	preurizedBprerakB
preparedneB	prepared.B	prepared,BpreminedBpremine,BpremiedBpremie.BprematurelyBpreident-elect,B
preidency.B
preidency,Bprei,BprefixB
preferred.Bpreference.BprefecturalBprefacedBpreethiBpreetBpreervedBpreervation.Bpreentation.BpredominantB
predicted:B
predicted,BpredateBprecribeBprecipitatedBpreciionB
precedent.BprecautionaryB	preading,B
preadheet.Bpread.Bpread,Bpre-publihedBpre-paidB	pre-orderBpre-manufacturedBpre-emptiveBpre-electionBpre-determinedBpre-conenu.Bpre-approvedBprcBprayingBprayerBpratBprao,BprangBpraiingBprahantB	pragmaticBpradeh.Bpractitioner.B
practicingB
practical,BpracticableBprachiBpraannaBpowerleBpower-that-beBpowell.Bpourebrahimi,BpounceBpouliotB	potulatedBpotulateBpottingBpothotB
potfinanceBpoter.BpotentBpotcoinBpotatoBpot-quantumBpot-production.B	pot-moneyBpot-covid-19B
portuguee.B
portuguee,BportrayBportman,BportmanB	portland,Bportico.comBportbookB	portable,BpornographicBpornhub.Bpork-barrelBporche,BporadicallyBpopupBpopcorn.BpopcornBpop-upBpoorer.Bpoor.Bpoon-fedBpoon,Bpooling.Bpoolin.BpoolinBpool.bitcoin.com,Bpool)BpookaliciouBpoofingBponyBponte,BpolyuBpolynexuBpollterBpolliBpolled,B	polkadot.B	politico.BpolitickingBpoliteB	politburoBpolipayBpolihedBpolicyholderBpolice)BpolarBpoland-baedBpokrandtBpokkt,B	pokewomanB
pokepeopleBpokemonBpoition:BpoiltB
poibility,BpoeedBpoe,BpocketedB
pocketbit,Bpoabit,Bpo,Bpnc,Bpm,BplynBplunkedBplunged,B	plunderedBplunderB
plummeted,Bplummet,BpluggingBplug.Bplug,Bplu)BpllcBplitterBplightBpliego,BpleardaoBpleaing.BpleaingBpleaed.Bpleae.Bplea.Bplea,Bplc.Bplc,BplazaBplaytation,B
playtationBplayon,BplayoffB
playgroundBplayer!Bplayed.B	playbook.Bplay?BplauiblyB	platinum.B	platform?B
platform),BplateredBplate.B
plantholt,B	plantholtBplantedBplantation,Bplan?Bplainly,BplainlyBplain,B	plagiary.B	plagiary,BplagiaryB
placement.B
placement,Bplaced,Bplace?Bpizza.BpixBpivx,BpivxBpitruzzelloBpit.Bpiru,BpirliadiBpiritu,BpirituBpirit.Bpirit,BpiratbyrBpiraledBpiracy.BpiracyB	pipermailB	pioneeredBpioneer.Bpinoff,BpinoffBpinkoinBpinella,Bpine,BpineBpindle,BpindleBpinchaBpinarBpin-off.BpillowBpile,BpilatuBpierBpiedBpie,B
pider-man.B
pickthumb,Bpick-upBpichaB
picetoken,B	picciott,BpicciottBpiarideB	phyician,Bphyical,Bphuket,BphraingBphotonicB	photohootBphoneyBphone?BphnomB
philoophieBphiloopher.BphillipeBphiherBphemexB	pharmcorxBphan,BphaingBphae;Bph/.Bph/,Bph.d.,BphBpfeffer,BpfefferB
pewdiepie,BpeverellBpeudonymityBpetronaBpetrodollarBpetro-peggedBpetraBpetitioningBpeteren,B
peterburg.Bpetco,Bpetalen,B	pervertedB	peruvian,BperueBperpetuatingBperpetuatedB
peronally,Bperonality.Bperonalitie.BperonageBperon()B	perniciouBpermit.Bpermit,B	permiive,BpermiionlelyB
permiible.BpermBperjuredBperit,B	peripheryB
performer.Bperformance-wieBperfect,B	pereveredBpereveranceBperenniallyBperception.Bperception,Bpercentage-wieBpeqtoB	pentagon,Bpent-upBpennylvania-baedBpenningBpenitentiary.BpenitentiaryB	penioner.Bpenion.BpenhBpendbchBpencilBpenalty,B
penalizingBpenaBpeloi,BpellfireB
pell-boundBpekov,BpekovBpeimitBpeimim.B	peercoin,BpeercoinBpeer.Bpeedy,Bpedn.Bpedn,BpeddlingBpeddledBpeculating.B	peculate.B	peculate,Bpectrum,B	pectator.BpecieBpecializationBpecial.Bpece,Bpec.Bpec,BpearheadingBpeaker.B
peacefullyB	peaceful,Bpeace;Bpe.Bpe,BpdvaBpdrBpdfBpccBpc.Bpbtc,BpbmtB	paytomat.B	payqwick,Bpayment-appBpayloadBpayjoinBpayid,BpayidBpayeeBpaycek,Bpaybutton.cahBpaybackBpay-to-cript-hahBpay-per-lat-n-hareBpaxo.Bpawned.Bpawned,Bpaulon,BpaulonB
paulo-baedBpauloBpatureBpattayaBpatrynB	patrioticBpatriceBpatient,BpatienceBpatezoneBpatebinBpatched.BpatcheBparticular:BparticleBparticipating,BpartianBpartedBpartakeBparole,BparlerBparkerBparingBparibaB	pardoningBpardonedBparatyB
paralympicB	parallel,BparaiticB
paraguayanBparafiBparadie.B
parabolic.BpapuaBpaport,BpapillonBpaphrae.BpaoBpanoplyBpannellBpannedBpankajBpanih-peakingBpanic-ellingBpaniardBpandemic-ledBpandemic-drivenBpancakebunnyBpammyBpammerBpam.Bpam,BpaltryB	palpable.BpalomaBpalleyBpalletBpalihapitiya.BpaletineBpalellaBpaledBpak.BpaivelyB
paionatelyBpaing.B	painfullyBpaiBpagni.BpagiBpag,Bpae.BpadBpackerBpacex.Bpace;Bpace:BpablompaBp2pkhBpBoziik,Bowner:Bowned.BowaBovrland.Bovrland,Bovr,BovexBoverwhelmed,B
overweightB	overview,Boverubcribed,Bovertock.com,B	overtock.B
overtated.BoverrunBoverlookingBoverlappingB
overhyped,BoverflowingBoveretimateB
overdrive,B
overcomingBovercollateralizationBoverchargedBoverbearingBouttripBouttanding.Bouttanding,B	outreach.Boutrage.Boutput)Boutpot,B	outlawed,BoutlatBoutide.Bouthwet,BouthdownBouthbankBouth-eatBouth,BouterB	outdated,Boutcome,BoutburtBoutboundBout-of-the-boxBouringBourelve,Boup,BoupB	ountokun,BoundetBoul.BouiaBou,BouBotto.Botto,BotoBother!B
otc:bfarf)BorwellBorrinBorrBormandyBorlando,BorlandoBorionx,Borigination,B
originate.Borganiation.BorelyBorehovBordonioB
ordinance.B
ordinance,B	ordering.Bordeal.BorbihBorbanBorbBorange,Boracle-baedBoptionalityBoptimumBopticalB
opprobriumBopportunity:B
opportunitB
oppoition.Boppoite,BopiumBopioid.BopinionatedBopholabBophiticated.B
operator).BoperationalizeB
operating,Bopening.Bopenea.B
open-worldBopen-endBopacityB	op_rhift,B
op_return.B
op_return,B	op_lhift,B
op_invert,B	oon-to-beBonuB	ontology,BontologyBontario-baedB	onnenheinBonlyfan.BonlycoinB	onlooker,BongxiuBongbirdBong.BonfoBonelifeBoneelf.B
onecoin.euBone-wayB	one-tenthBone-offBone-dayBoncomingB
once-a-dayB	onatibia,Bon-horeB	on-chain,Bon-boardBompoBomiion,B	omewhere,BomerBomenic,Bomen.Bomehow,B	ombudman,BolverBolvency.Bolved,Bolve.Bolve,Bolution?Bolution:BolticeBollieBoliver,B	oliveira,B
oligarchy.B
olidifyingB	olidarityBolicitation,BolekiyBolekandrBoleimaniBoldier.Bolder.BolayinkaB	olatunji,Bolar,BolajideBolafBolactiveB	oktyabrkyBokloBoitaBoinbajo,Boil-baedBohrabBohnickBohiocrypto.comBohanian,BofterB
oftentime,BoftenedB	oft-citedBofitelBofficial-lookingBoffhore.B
offering).B
offering),Boffered.B	offenive.Boffence,B	offchain,Boff:B	off-ramp,B
off-limit.Boff-gridBoff-exchangeBofa.BodomBodeaBoctogenarianBoct>63k,Bock.BociopathB	ocio.com.B	ocio.com,Bociety;Bocietie:B	ocializedBocean,B
occurring.Boccurrence,B	occurred.BoccuredBocc,Bocar-winnerBocar,Bobtain,Bobtacle.Bobrador,Bobolete,B
obolecenceB	oblivion.B	oblivion,B
obliterateBobligingBobligateBobjectedBobject,BobitB	obfucatedBoberve.Boberve,BoberanoB	obcurity.BobcurityBobcuringBobcuredB	oaka-baedBo1ex.Bo-ytemBo-1BnytBnydig.Bnyc,BnyanBnyaBnxtB	nwaniobi,BnvtweetBnvidia.Bnvda),Bnv,BnuttyBnutBnurievaBnureryBnuneBnumerology,B	numberingB	numbered.Bnumber:Bnuland,BnulandBnuianceBnubitBnub,BnuancedBntuBntfBntaBnppaBnpoBnowotnyB	nowoenetzB	nowglobe:BnowglobeBnow-retiredBnow-deletedBnow)Bnovy-williamBnovakBnov>98k,BnouveauB
notradamu,Bnotion,BnoticiaBnoticed.Bnotice:Bnotch.Bnotary.bitcoin.comBnotalgicB	notalgia:Bnotability.B	notabene,Bnot?Bnot:Bnot-o-ditantBnorway-baedBnorthernmotBnortheaternBnorthbound,Bnorth.BnormundBnormal.Bnormal,BnoonB
nonviolentBnontopBnontechnicalB
nonexitentBnon-violenceBnon-trut-baedBnon-traditionalBnon-technicalBnon-taxBnon-regulatoryBnon-qualifiedBnon-optional,Bnon-monetary,Bnon-martphoneBnon-interactiveBnon-idBnon-exitentBnon-exchangeBnon-currencyBnon-corporateB
non-brokerBnon-aggreionBnominee,Bnomination.Bnomi,BnomenclatureBnokia.Bnok,BnokB	noie.cah,Bnoie.BnoguchiBnoedivedBnoediveBnodecounterBnode.jBnoda,BnocoinerBnobody.Bnoah,Bno-confidenceBno-brainer.Bnl,BnlBnjorogeBnjBnittyBnirvanaBninth-largetBnintendoBninja,Bninetie,B
nineteenthBnine,BniklaBnikkei,B
nikiforov,B	nikiforovB
nightmare.Bnight:BnieriBnicollBnicker,Bnicehah,B
nicaragua,BniBnguyen.BngdcBnft1Bnft.gametop.comBnft-relatedB	nft-carceBnext?Bnexi,B	newworthyBnewwireBnewport,BnewportBnewpeakBnewpartB	newcomer.B	new-foundBnevilleBnevi,Bnevada.Bnevada-baedB
neutralizeB
neutralityBneuralBneu-ner,Bnetwork-poweredBnetwork-levelB	network),B	netwalkerBnettingBnetleBnetflix.Bnet-flowBnet-avvyBnerdyBnerdBnepaleeBnepal?Bnepal,Bnelon,BnejcBnegocio,B
negligenceBneglect.BneeredBneerajBneedleB	neceitateBnebulou,Bnear-failedBneakerBneBnduom,BndicBncty)BncbaBnbuB	nazionaleBnazarov,BnazarovBnayayer,Bnavy.Bnauticu,Bnationwide,Bnationally,BnationalitieBnateBnatbankBnarvaezBnarratedB	narcotic,BnappyBnappingBnaple,Bnaphot-cloneBnapbackBnaokazuBnanopaymentBnangengBnanenBnandlalBnaming-rightBnamibia,Bnamed.B	namecheckBnameakeBname-calling,Bnam-ki,Bnam-kiBnamBnakeB	nak-yeon,Bnak-yeonBnairobi-baedB	naira/btcBnailwalBnadzoruBnacomB
nabiullinaBnaa,Bna?Bna,Bn570Bn.v.Bn-1aBmytificationBmyticalB	myterium,Bmyterie.BmypaceBmyetherwallet,Bmyelf,Bmyanmar.Bmy!BmwedziBmw.Bmw,BmutuallyBmuterBmutedBmut,BmurungaBmurphy,BmurgioBmurder.BmuratBmural,BmuralBmurai,Bmunicipality.Bmunicipalitie.Bmunicipalitie,Bmunich,BmungaBmundoBmundialBmumbai.Bmultiplied,Bmultiplayer,BmultiignatureBmultiig,BmultifacetedBmultidiciplinaryB
multichainBmulti-platinum-ellingBmulti-platinumBmulti-platformBmulti-nationalBmulti-millionaireB	multi-ig,Bmulti-factorB	multi-dayBmukuruBmuhroom,BmuhroomBmuhak,BmuggleBmugabe,BmugabeBmuffinBmud-linging,BmucularBmuch?B
much-oughtBmucat,BmuaBmtrycz,Bmtr),Bmtn.Bmt5BmpcBmp,BmovitarBmoving,Bmouth.BmourningBmoura,BmountedBmount,BmotivationalBmotivateBmotifBmotiBmotherboard.Bmotherboard,Bmotepe,Bmotel,BmortyBmortenBmortalBmorphBmorocco,BmoreyBmoreoverBmooth.Bmooning,BmooningB	moonbeam,B	monument.BmontrouB	montroll,BmontrollBmontreal-baedBmonth-over-month.Bmonth-on-month,B
monter.comBmonteBmontana.BmontageB
monopolie.B	monologueBmonobankB
monitored.Bmonitor.Bmonitor,Bmoniker.Bmoniker,Bmonie,BmonieB	mongo-db.BmonfexB	moneytechB
moneygram.BmoneyedBmoneycontrolBmoneybooker,Bmoney),BmondruBmonarchBmomentarilyBmomaaBmomaBmollyBmoller,BmoldovanB	moldavianBmokingBmokeBmoieev,BmohitBmohenB
mohammadi,Bmohammad-rezaB	mohammad,Bmogul,BmogomortgageBmofBmodule,Bmodi.B
moderationB
moderatelyBmochiB	mobilizedB
mobilecoinB	mobikwik,BmobikwikBmoad,Bmnuchin.BmnoB	mnemonic,BmnbBmmoBmmmBmlb,BmlarBmkBmiyamotoBmixtureBmixnetBmixer/tumblerBmixed,BmitrutBmitmB	mitigatedBmith.BmitchellBmitakingBmitaken.BmitakenBmit.Bmirziyoyev,BmirokuBmirnovBmiriBmirepreentationBmirceaBmiraBmiqB	miplaced.BminuculeBmintpal,Bminting,Bminted,B	mintable,B	minoritieBminnowB	minneota,BminkabuB	minitrie,BminitBmining:Bmining.gBminimum.BminimaldataB
mini-printBmini-poBmini-documentaryBmingxingBmineralBminer-fundedB	minecraftBmindetBminder,BminatiBminableBminBmimblewimble-baedBmiloB	million],Bmillion-dollar-pluBmilliecond.Bmillibitcoin,B	millibit.Bmillennial.B
millennia,B	millenialBmill.BmilkyBmilkenBmilkBmilitiaB	military.BmilingBmileadBmile,BmildlyBmikanBmiionaryBmiion:Bmiion,Bmiing,B
miinformedBmiinformation.B
migration,BmigivingBmightyBmight.BmifitBmierablyBmie.BmidtermB
middle-menBmiddayBmid-to-lateBmid-tierBmid-termBmid-may.Bmid-juneB	mid-july.Bmid-julyBmid-januaryB
mid-augut.B
mid-april.B	mid-2017.Bmid-2016B	mid-2010.Bmicrotranaction,B	microtateBmicrooft-ownedBmicro-tippingBmicro-blog,Bmicro-bitcoinBmicro,BmicheleBmichalBmichael,B
micarriageBmicahBmicaBmicBmiappropriationBmhzBmgtBmfbBmfa,Bmetky.BmeticuloulyBmethodologieBmethodicallyB
methodicalBmetavereme,B
metaveremeB
metatraderB	metatableBmetaphorBmetallaBmetaBmerkleBmerger.B
mercifullyBmerchB	mercenaryBmercede-amgBmepBmeopotamianBmenu.cahBmentalmarketB
mentality.BmeneBmendezBmenaceBmen.BmemorizeBmemorandum,Bmemorabilia.BmemorabiliaBmemopayBmemoir.Bmeme-currencyBmember-countryBmellon)BmeilichBmehrBmeghanBmegawayBmegahahB
mega-whaleBmega-miningBmeg,Bmeet-upBmedievalBmedici,B
mediation,B	mediationBmediatedBmedia:BmedellB	meddling.Bmechanicville,BmechanicvilleB	meaurableBmeat,B	meari.io,B
meaningle.BmeaningfullyBmealwormBmeager,BmeagerBmeagedBme?BmdtBmcoBmcmahon,BmclemoreBmckibbinBmchenry.Bmchenry,BmcgrathB	mcdonald,BmcdB
mcculloughB
mccormack,B	mcafeedexBmcafee:Bmboweni,BmbkBmbaBmb)BmazeBmayweather.Bmayor-preidentBmayfairBmaximum-ecurityBmaximeB
maximalim.B	maximalimBmavi,Bmaven,BmavenBmauriceBmaturation,B	matumoto,Bmatuda,BmattreeBmatthiaBmatrix.Bmatoni.BmatoBmatic.BmatiB
materpieceB
matermind,BmaterializingBmater,B	matchpoolBmatche.Bmatch.Bmatch,B	maryland.BmarvinBmaruiBmartkey,Bmartin,BmartelBmartbch,Bmart-contractBmarlton,Bmarlin.B	markovichBmarketwatch.Bmarket-relatedBmarket-leaderBmarket-baedBmarinaBmariahBmariBmarhaBmarhBmargaretBmarchedB	maracaiboBmar?B	maqueradeBmanxBmanure.BmantleB	manpower,BmannyBmanipulated,Bmanion.B
manhattan,BmanfredBmandelaBman:Bman-in-the-middleB	mamutual,Bmalta-regiteredBmalpracticeBmalmiBmalloukB
malicioulyB
maldonado,BmalayBmalavikaBmakvere.com,BmakinBmakhloufBmakerplace,B	makerdao,B	makeover:Bmak,BmajorlyB	majority.Bmajli,BmajliBmaively.Bmaive.BmaionB	maintreetBmaintainer.Bmaintained,B	maintain,BmailmanB
mailchimp,BmailboxBmail,BmaijoorBmahmud,BmahmudBmahhadiBmahed,B
magnitude.Bmagnifymoney.comBmagnate,Bmagic,B	madoguchiBmadne.Bmadne,BmacyBmacro,BmacrinaBmacriBmacoBmacbookBmacau.Bmacau,BmacakillBmaayohiBmaaoBmaachuett-baedBma-producedB	ma-marketBm2,Bm1,Bm0,Blyu.Blyu,BlyricBlynch,BlydiaBlxdxBluxuriouBluthfiBlurkBlundBlumpurBlumpingBlumino.Blumber.Blull.Blull,Blukka,BlukewarmBlukejrBluhnikovBluh,Bluggih,BludicrouBluczynkiBlucky.Blucky,Bluck,BlpvhBlp,Bloyalty.Bloxton,BloxtonBlowneBlowdown,Blow-reolutionBlow-cot,Blovenia-baedB	love-hateBlovakia,BlovakBlourdeB	louiiana.Blottery,B	lotterie,Blorax,BlorBlopp.BlootingBloot.B	loopnrollBloopBlooneyBlooming?Blooming.Bloom,Blookout,BlooBlonnieBlongedBlong-windedB
long-term,B	long-onlyBlong-etablihedBlonelyBlogo,B	logiticalBlogitic,Blogical,BloftBloer,B	loeffler,BloefflerBlodgedBlodgeBlodderB
locomotiveBlockyBlocktripBlocker,BlockerBlockboxBlock-upBlock-in.BlocalizationBlocalethereumBlocale.Blocalcrypto.comBlocal.Blobbyit,Blobby.BlobbiedBlobban,BloathingBloathe.BloatheBloaningBloanedBloadingBlnd.B
ljubljana.Bliwa,B
livington,BlivetreamerBlivetockBlivera,BlivenBliting:BlitigantB
literatureBliteraryBlite.im,BlitanyBlira,Bliquidator,Bliquid-cooledBlippage,Blip?BlinoB	linkedin,BlinkeB	linguiticBlingham.BlincolnBlin,B
limelight,Blimbo.BlimboBlimaBlim.Blilium,BlikingBlikewieBlikely,Blike:B	like-kindBlik.B	lightyearB
lightpeed,Blightning-tyleBlightning-fat,B	lightnet,Blightly,B
lighthoue,B	lifetyle,Blifepan,B	lifecycleB
life-bloodBliechtenteinicheBlie:Blick,B	licenure.Blicene)Blibya,Blibor,Blibon,B	libertyx,B	libertie.Blibertarianim,BlibertarianimBlibertaB
liberland,Bliberation.B
liberationB	liberatedBliberateBliberalizationBlibanB	liaoning,BliangBlgBlezhavaBlevy,Blevie.BleverBlevelingBlevB	leuthing,Bleuth,B	leukemia.Bletterhead,Blerner.B
leperance,Bleone,BleonardBlentBlenientBlendedu,Blen,B	lemonade.BlemonadeBleicnikB	leibowitzBlei,BleiBlehnerBlegworkBlegoBlegit?Blegit-lookingBlegionB
legilator.Blegalization.B	legality.Blegacy,Bleg.Bleg,B	left-overBleeve.BleetjaBleekBledwaba,Bledn,BlednBledger-baedBleclairBlebanon.Blebanee-americanBleave.B	learning.Blearned.Blearn.Blearn,BleapingBleapfrogBleapedBleaker,Bleaked,BleahBleafletBle)BlcdBlcB
lazyfox.ioBlayperonBlayout.Blayoff,B	layer-twoBlawonBlawnBlaw?BlaveryBlave.BlaveBlaurel.BlauranceBlaundering)B
launderer.B	laundererBlaunchpool.B
launchpoolB
launching.BlaughterBlaughingBlauBlatvia,BlatrootBlatet.BlateralB	late-2017BlatchingB	lat-ditchBlarger-than-lifeBlara,BlaraBlanka,B	languihedBlaneBlandown,BlandownBlandonBlandlideBlandau,BlanceBlampBlammingBlamidoB	lamented.Blamborghini.Blambino,BlambinoB	lakhanpalBlakeideBlagoBlagarde.Blag,BlaforgeBlaerreBlaer-focuedBlaebikanBladieBladenBlacking.Blacher,BlacedBlabourBlabergeB	labeling.Blabel,B
l2fee.infoBl.p.Bl.a.,Bl-btc,BkywardBkyuhuBkyrocketing,B
kyrgyztan.BkyrgyzBkyodoBkyiv,B	kycraper,BkycoinBkyc/ofacB	kybridge,Bkyber,Bky,BkwonBkvahuk,Bkuwait,BkuvandzhievBkurdBkuo,Bkuni,B	kumbhani,Bkumamoto-energyBkuiBkuhneBkudiBkucoin.BkubitxB
kubernete,Bkryptovault,Bkrw,Bkrug,BkronorBkronoBkroner.Bkrona,BkritinBkriBkraftBkraBkpBkothari,BkothariBkorobogatovaBkornBkoovoBkoo,BkonzeptB	konovalovBkonkinB
kommerant,BkolinBkokeh,BkoinfoxBkohenBkodriBkodaq-litedBkodaq,Bkodak-brandedBkodak,BkociBknutli,Bknock-onBkniveBknickBknew,B	knee-jerkBkneeBknackBkmdBkmBkleroBklagge,BklaBkktB	kjartanonBkiwieBkiwiBkivirBkittyBkittie.Bkittie,Bkitten.Bkitt,B	kitco.comBkitchenBkirmiheBkirkhornBkirkB	kirdeiki,BkippingBkippedBkingpin,Bking:BkimbrelBkiltBkilowatt-hour.Bkilowatt-hour)Bkilo,BkikvadzeBkiklabbBkihBkiev.Bkid,B
kicktartedBkickoffBkick-tartingB	kick-tartBkhyberBkholaBkhanna,Bkhaled,BkgilBkftcBkeytore.BkeycardBkey;B	key4coin.Bkey),Bkew.com,BkeralaB	kepticim,B	keptical.Bkeoken.Bkenyan.Bkeno,B	kenicoin.BkenangaBkelo,Bkelly.BkeletalBkeithBkeier.Bkeep,Bkeene;BkazukiBkaunB	kauffman,Bkattetyrelen,BkathBkateboardingBkarpelBkarlonB	karlberg.BkarlbergBkarenBkaraBkapoorB	kapilendoBkapfidzeBkaparovBkantonalbankBkanoBkanazawaBkamilBkamenBkaliningrad,B
kalahnikovBkakao,BkaireddyBkahmirBkahminerBkahleBkahindiB	kafkaequeBkaeyaBkaboyoBk.im.B	juxtapoedBjuungle.net.BjutuBjutbetBjurrienBjuniorBjungleBjunctureBjunckerBjump.Bjump,BjulioBjuiceB
judgement,BjudeBjtolfi.Bjr,B	joytream,BjoyceBjoy.BjournaliticBjorgenenBjordaan,BjordaanBjong-unBjonaBjokinglyBjoke;Bjohnon.Bjohn,Bjohanneburg.Bjohanneburg,BjohanneBjoefBjoble,BjobchainBjj.Bjj,Bjiu-jituBjitterB	jiratpiitBjinyrulBjinping.Bjin,BjinBjibeBjiangxiBjhunjhunwala,BjhunjhunwalaBjhBjeu!BjeterBjet.BjerualemBjeopardizingBjemmaBjeieBjeffrieBjeduorBjean-claudeBjean-baptiteBjeanBjd.com,Bjd.comBjdBjcBjaxx,BjavedBjava,B
jaravijit,B
japan-baedBjapan-backedBjanenBjame.Bjame,Bjam:B
jam-packedBjakarta.BjakartaBjailbreakingBjaggerBjae-inBjackonvilleBjacking.BjaanBj5BizabellaBiykeBix-year-oldBix-yearBix-hourBivypay,B	ivendpay.BitvanB	ituation:Bituated.Bito,BitoB	itineraryB
itharaman,BitchingBitalian,Bit)BirwinBirvingB	irruptionBirreverible,BirreponiblyBirreitible.B	irregularBirreconcilableBironx,BironxBironfxBirmaBirleaf,BiriuBireneBircBiraq,BiraminerBipadBiovBiota.BiotBiorio,BiolationBiolated,Biohk.Binvoked.BinviiblyBinvetor?Binvetor)B
invetment:BinvetimentoBinvetigated.Binvetigated,BinvetcoB	invet.comBinverelyB
inventory,BinventBinvaion:Binvader.BinurmountableBinurer.Binufficient.Binu.BintruderBintruction.BintrinicallyB	intraday.Bintra-africanBintitutionalizedBintitutionalizationBintitutional,B	intitute:BintilledBinterviewingBinterventionitB
intervenorB
intervene,B	interval,BintervalBinterrogation.BinterpretiveB	interpretB	interpol,Binteroperability.Binteroperability,Binternet-baedBinternacionalBintermediation.BintermediationBinterference.B
interferedB
intereted.BinterectBinteramericanBinter-vehicleBintenified.Bintenified,Bintene,Bintelligently,Bintel.BinteaBintantaneoulyBintant,Bintance.BintallerBintallation.BintakeB	intagram.BintactB
intabilityBinquiry,B	inquiringB	inquirie,BinquireBinpectorateB	inpector,B
inpection,B
inolvency,B
innovator,Binnovation?B	innovate.Binnoilicon.B
innocence.Binno3dBinnateBinnBinlandBink.Bink,Binjury,B
initially.BinitenceBinited:Binit:BinignificantBinheritance,B
inhabitingBinha,B
ingularityB	inguhetiaBingletonBingleeedBingle-familyBingle-entryBingle-digitB	ingle-dayBinglBinger.B	infuriateBinfura,BinfratructuralBinfoyBinfowarB	informed,B
informant,Binfobae,Binfluential.B
inflectionBinfetedB	inferenceB
infeaible.BinfanteBinevitability.Binequality,Binefficiency,Binefficiencie.BinecureBineBindutry;Bindutry-pecificBindutrializedB
indutrial:Bindutrial-gradeBindutrial-caleBinducingBindoctrinatedBindividuallyBindividualitBinditinguihable,B
indiequareB
indiegogo.BindieBindictBindication,Bindia?BindexingBindex:B
index-baedBindependently.Bindependence,B	indemnifyB	indeliblyBindefinitely,B	indebted,BincubateBincredulityBinconvenience.Binconvenience,B
inconitentB
incongruouBincomplete.BincompetenceB
incluiveneBinclude,Bincline,BinclairBincitingB	incidenceBincheonBinchB
incentivieB
incentive,Bince-deletedBincarceration.Binc.)Binc).Binc)Binappropriate,BinappropriateBinadvertentlyBinactaBinaccurate,Binaccuracy.Binacceible.Bina,Bin?Bin:Bin-platformBin-kindBin-homeBin-fighting,B	in-depth,B
in-betweenB	imulator.B
imulation,Bimtoken,B	impunity.B	imprudentBimprobable,Bimprionment,B
impreivelyB	impreive.BimpregnableBimpracticalityBimporterBimpon:Bimpoed.B	imploringB	implored:B	imploion:BimploionBimplificationB
implicatedB	implicateBimplex.Bimplex,B
implement.BimperonatedBimperceptiblyB	imperatumB
imperatrizBimperative.BimpedingB
impedimentBimpae.B	impacted,BimonettiB
immutable.Bimmutability,BimmutabilityBimmortality,B
immigrant,BimmeriveBimmereBimmerB
immediate,BimmatureB
imjacking,B	imjackingB	imitatingBimitateBiminer,BiminerBimilar,Bimf.BimdbBimamBimagine,B
imaginableBimageryBim-wapping,Bim,BilyayevBilviuBilvioB	ilverman,BilvaBiluanov,BiloedBillutrator,BillutrationBillutratingB
illutrate.Billinoi.B
illicitly.Bill-preparedB	ill-fatedBilkroadBiliaBilent.BildemaroBilbo,BilanderBilanBilamBikurBijBiii.BihikawaBihenyen,B
ignup.cah,Bignup.BignorantB
ignorance,Bigning,BignifieBignificance.B	ignatova.BignacioBigmaBighah.BighBifwalletBiftingBifp-freeBif.BiegelBiege,Bidle.BidioyncraticBidingBidg,BidfcBideway.Bideway,B	ideologitB
identitie.B
identitie,Bidentifier,Bidentified,B
identical.Bidemen.Bidehift.ai.Bideg,BidedBidealimBidea:BicyBiconomiBicon,Bico?BiciciBicfBiceland,Bice3Bice,Bicahn,BicBiboBibkr),Bibinex,BibcBibarraBibaBiamBiakBi.e.Bi.eBi!BhyundaiBhyteria,BhypothekarbankBhypocriyBhypocriticalBhyperinflatingBhyperbitcoinization:Bhyperbitcoinization,Bhyper-realiticBhype?BhyoungBhyogoBhydroelectricity.Bhydro-poweredBhydro-powerBhydra,B	hyderabadBhycm,BhyamBhwBhvetov,Bhuvalov,BhutteredBhutchin,Bhut8Bhut.Bhut,Bhut)Bhurdle.Bhurdle,BhuntorBhung,Bhundred,BhumpBhumor,BhummingBhumilityB	humanity.BhumaneBhuman-friendlyB
hullabalooBhulginovBhuge.Bhuffled.BhueinBhud.BhuBhtml5Bhtml,Bhtf?BhtfBht.BhtBhryvnia,BhrunkBhrinkageBhrink,BhrimpyBhrimpBhrankBhowmuch.netB	howeycoinBhoweyBhoweredBhowell,B	howcaing,Bhow]B
houmanadr,Bhoue:Bhotz,BhottieBhotterBhotpot.Bhotline,Bhoting.Bhotile.Bhotcake.BhotbitBhortlinkBhortedBhortcut.Bhortcoming,BhorrorB	horrendouBhorizen,BhorizenBhorcanBhoppedBhopkin,B	hopepage,Bhoped.BhooplaBhoo.com.Bhonubi,BhonoredB	hong-kongBhoneypotBhonet.BhonenBhonda,B	homicide,B	homepage,B	home-baedBholographicB	holmelandBholmeBhollowBholiticBhole.Bhokinon.BhohoBhofmann,BhofmannBhoeinBhoe,BhoeBhocBhobbyBhoax.BhoaxB	hoarding,BhoarderBhmuelBhmt.Bhmrc.BhkmaBhk,BhivrB	hivemind,BhivemindBhitpoterBhitory:Bhitorie,BhitmanBhitlerBhithertoBhitcoin.com,Bhitcoin.comBhitcoin,Bhitch.BhirtB	hipchain,BhipchainBhint,Bhinoda,BhinichiBhine.BhimeBhilton,BhillierBhiller,BhildaBhikingB
hijacking,BhiharaBhighwayBhighet-paidB	high-peedBhigh-growthBhigh-denity-loadBhigginBhifted,BhideoutBhidekiBhidBhicoxBhibeBhib.Bheyday,BherzogBherzegovinaBherronBheroe,BherlockB	heritage,BheriffBhere;BherderBherdedBherdB	herculeanBherbertBherald.Bher,BhepardBhenzhen.BhenyepBhenovaBhenneyBhence,B
henanigan,B	henaniganBhenanBhemp.Bhemmati,BhemleyBhelveBhelter.Bhelter,Bhelm.B	hellhole,BhelioBhelinki.BhelinkiBheldonBhelbyB
heitating.BheitateBheit.BheireBheinzBheinrichBheinouBheineB
heindorff,B	heienbergBheidiB	hegemony,BhefnerBheffieldBheel:BheehaBhedging,BhedgeyeB
hedgefund.BhedgedBhebei,Bheavy,Bheavily.BheavierB	heartlandBhearthtone,B
heartbreakB
heartbeat,Bhear:BheapB	healthilyBhealth.Bheaded.B	headcountBhdrBhcheltinBhbo.BhbBhazard,Bhayek,BhawkihBhawaii.Bhaving.Bhaving,BhauntingBhaul.B	harvetingBharvard,B	harrowingBharrapBharpetB	harpeningBharneeBharmony,BharleyBharkerBhariramani,B
hariramaniBhariaBharherBharetipBhareholdingBhareableBhardhip,Bharder,BhardenBhard-hittingBhard-hitBhard-forkingBhard-forkedB
hard-codedBharare.Bharare,Bhappine.Bhappen?BhapleBhapiro,Bhapehift.ioBhape.Bhape-hiftingBhaohanBhankyungB	hankyorehBhankarBhangzhou-baedBhangzhouBhangoutBhanghai,BhaneBhandy.BhandongBhandomeBhandheldB	handhake.B	handhake,Bhandcah.BhandbookBhand:Bhand-onBhancockBhana.BhanBhamptonBhammeredBhamingBhamedBhame.Bhama.BhamaBhalving:B	halveningBhalved.Bhalted.B
halloween,BhalfwayB	half-hourBhale.BhalcahBhakimi,BhaircutBhair.BhaidianBhaichaoBhahed,BhahcowBhahchainBhahamB	hah-powerBhah),Bhague.Bhady,BhadleyBhade.BhadderBhad.B	hacktivitBhackreactorBhackleB	hackeroneB
hack-proofBhabitatBhabit.BhaberBhaanxiBhaaBha256.Bh.i..Bh.e.Bh.b.Bgym,BgyeonggiB	gyeongangBgwynethBgwaiBgutoBgutmanBgurleyBgurbacBgunnerBgunhotBgun.BgumtreeB	gulftreamBgulf.Bguizhou.Bguizhou,Bguindo,Bguggenheim,Bguework,Bguevara,BgueeBgue.Bgue,Bgud.BgudB	guatemalaBguardingBguardiaBguard,BguaranteeingBguaranteed.Bgt,BgrupoB	grunzweigB	grovelingBgroup;B	groundhogBgroundedB	ground-upBgroomingBgroceryBgrocerieBgrobler,BgrittyBgrippingBgrippedBgrinch,Bgrim.B	griffith,B
grievance.Bgrider,Bgrew.BgreetingBgreenproB
greenpeaceBgreen-lightingBgreen-lightBgreedy.Bgreedy,BgreedyBgravitatingB	gravitateB	gratuitouB	gratitudeBgraph,BgrapeBgrandparentBgrandmotherB	grandhoreB
grandchildBgramercyBgrainB	graffiti:Bgraff,B	graduatedB	graduate,Bgrade.BgrabbingBgrab.Bgpu,Bgoxx,BgoxdoxBgovernment-upportedBgovernment-ponoredBgovernment-authorizedB	governed.BgovcoinBgougingBgotten.BgotoBgotatohi.comBgormanBgorillaBgorgeBgooeB
goodlatte,Bgood-heartedB	gonzalez,BgonzalezBgonghengBgomezBgolix.BgolfB
goldmoney,Bgoldfoundinh*tBgoldbugB	gold-likeBgogo,B	godfatherBgoddeBgocBgoat,Bgoal:BgoadedBgo-pay,Bgo-jekBgnomeBgniBgnbB	gmo-z.comBgmiBglowingBglove,Bglory.Bgloom.Bglobenewwire.comBglobant,BgloatingBglitch,BglimmerBglideraBglgBglenmedeBglenBgleeBgledhillBglazyev,BglawareBgladBglacialBgizmodo,BgizmodoBgiving,B
givecryptoB	giveaway,Bgive,Bgiutra,B	gitcah.ioBgirlfriend,B
girlfriendBgioraBginger.BgimmerB
gillinghamBgiguiereBgift.bitcoin.comBgieeckeBgiddyBgibon,Bgibb,BgibbBgiacomoBghonBghilaineBgharegozlouBgeture,BgetureBgethBget.Bgermany-baedBgergely.B
georgieva,Bgeorgia.B
geopoliticB	geometricBgeolocationBgeohot,BgeoffreyBgeoffBgeoeconomicB
geocachingBgeo-locationBgenler.BgenimouBgenerationalBgenerating.B
generally,BgeneraleBgeneral-purpoeBgeekyBgeekedBgeek.Bgeek,Bgchq,Bgbx,BgbbcBgb.BgazpromneftBgazprombank,BgazillionaireBgazeta,BgawaxabBgaw,Bgaurav,BgauravBgateway.cah.Bgate.Bgartner,BgarlandBgarg.BgardaBgarageB
gappelbergBgaoline,BgantzBgangter.Bgang.BganehBgananciaBganBgamifiedBgamification,Bgametop/walltreetbetB	gameplay.B	gameplay,BgamehowBgame.bitcoin.comBgamble,BgambleB	gamaroff,BgaltBgalperinBgallippiB	gallerie,BgallantB
gallagher,BgalitBgalaxy.Bgal,BgalBgakutoBgained.BgaffeBgadget.BgadflyBgackt,BgabrieleBgaborBgabdullina,Bgab,Bg-20Bg$B
fyntegrateBfxtb,BfxproBfx,Bfuturit,BfuturitBfuture:BfuturamaBfutu,BfutileBfuryBfurthetB	fungible,B
fundraier,Bfund;Bfunctioning,Bfunctional,Bfully-regulatedBfully-inuredBfull-reerveB	full-pageBfull-featuredBfull,B
fulfilled.BfujimakiBfuente,BfuellingBfuel,Bfued.Bfud.Bft.Bfry.Bfrutrating,B
frutratingBfrumBfrownedBfrownBfrotaB	frontier.Bfront.BfrolovB	frivolou,Bfringe.Bfrii,BfrighteningB	friendly.B	friendly,B
friendlineB	friedrichBfriday?Bfrick,BfrexitBfrench,Bfreire,BfreireB	freiberg,BfreiBfrehmanBfreethinker,Bfreeman,Bfreely.Bfreely,BfreelancingB	freefall,B
freedomainBfreedom:B	freedman,BfreedB	free-tierBfredrikBfreddyBfreakingBfraudulent,B	fraudter,BfranticB
frankfurt.Bfrank.Bfranico-baedBfrancoiBfranckBfranci.Bfrance?BfranBframedBframe.B	fragilityBfraer-jenkinB	fracturedBfractionalizationB	fraction,BfpfBfoxconn,Bfourth-rankedBfourteen-year-oldBfountainhead.BfoundrieBfoter,B
forward31,Bforum:Bforuall,B
fortnight.BfortitarB
forthrightBforthcoming.B
formulatedBformula,Bforming.Bformer.Bformer,Bformed.B
formalitieBforkerBforked-coinBforkatBforideB
forgotten,Bforgery,B
forfeitingB
forfeited.B	foreword.B
foreigner,Bforeign-baedBforeign,B	foreight,BforegoneB
forecater,B	forecaterBforeawB
forcefullyBforcefulBforce-cloingBforbe.Bfor?B
footprint,B	football,BfoolihB	fonacier,Bfomo-fudB
following:B	follow-onBfold,BfoiledBfoiaBfoe,BfocalBfoamingBfnbBflux.Bflux,BfluidlyBfluhBfluffypony,BfluentBfluBflownBflowee,BfloutingB	flourihedBflounderingBfloor.Bflood,BflirtingBflirtedB
flirtationBfliptarter.cahBfliptarter.Bfliptarter,Bflippening.B
flippeningBflip,BflintBflickBflexBflemingBflehBflawlelyBflat,BflahyBflahingBflahedBflagrantBflabbergatedBfizz.Bfixed.Bfixed-incomeBfivebuck.com:Bfive.B	five-foldBfiureBfiuBfitzpatrick,B
fitzgeraldBfittetBfitfightBfirt-of-it-kindBfirouzabadi,B	firewall,BfirefighterBfirearm:BfiranoBfiraBfioreBfinopoliBfinnBfinman,Bfinland-baedBfink.Bfink,BfiningBfinihed,BfiniheBfingerprintingBfinfetBfinding?Bfinder.Bfinch,BfinchBfinanzdientleitungaufichtBfinantilynet,B
finanowegoBfinanciero,Bfinancially.BfinancializationBfinancial-techBfinanciaBfinalityBfinaleBfinal,Bfilter,Bfilled.Bfill.Bfiling:Bfiler.B	filecoin.BfilanBfih.Bfigurehead.Bfight.Bfifth-rankedBfiercetB	fictitiouBfico,Bfiat:Bfiat-dominatedB	fiat-baedB
ff1achineeBfever.B
fetivitie,BfetchedB	fetch.ai,Bfertilizer.BferrariB
fernandez,BfernandBfendingBfenceBfelton,Bfelt.BfelineBfeet,Bfeel.BfeederB	feedback,Bfederally-regulatedBfedcoin:BfecB	february:Bfeature;Bfeather,Bfeat.Bfeat,Bfeaible.BfeaibleBfdleBfdBfciuBfccBfca.Bfc,Bfbk,Bfbi?Bfbi.BfbgBfawningB
favourableBfavor:Bfault,BfatwaBfatet.Bfatet,Bfate,BfatblockBfatal,Bfat-trackingBfat-foodBfarzamBfarming.BfarkaBfare.BfarageBfar:BfandomeBfandomBfanboyBfanaticBfamineBfamerBfame.BfalzonBfallout,Bfalling.B	fallaciouBfalconxBfakingBfaketohiBfaked.BfaithfulBfaith.Bfaint-hearted.BfahionedBfagundeBfactualB	factorie.BfactoidBfaction.Bfact:Bfacing,BfacinateB
facility),BfacilitatorBfacilitated.BfacialBfacadeBfabricationBfabricatingB	fabricateBfa-approvedBfa,Bf***BeyffartBeyedB	eychelle,BexxonBexx.B	extropianB	extremim.Bextreme,BextravaganzaBextravagantBextraterretrialB
extranonceBextraction.BextolledBextinctBextent,B	extenive.Bexpropriation.Bexpoure,Bexport,Bexponential,B	exponent,B	expoitionBexpobankB
exploited.Bexploitation.B
exploding,B	exploded.BexplanatoryBexplaining.Bexplain;Bexpire,B	expertie,Bexpert:BexperientialBexperience!BexpendBexpedition.B	expeditedBexpediteBexpect,B
expatriateBexpatB	expanded.B	expanded,Bexotic,Bexmo,Bexile.B
exhumationBexhilarating,B
exhautive,B	exhautingB	exhauted,B	exerciingBexercie,B
exemption,BexemptedBexempt.B	executed.BexcuedB	excrementBexcluively,B	excluive:B
exclaimed:B	exclaimedB	exciting!B	excitedlyBexciteBexchanging,B
exchangifyB	exchange]Bexchange-relatedBexcept,B
excellenceB	exceivelyBexceeBexanteBexaminerBexahah.Bexact.Bexact,BexacerbatingB
exacerbateBexacaleB
ex-centralBewingBewaldBevolved,BevillaBevil,Bevident.B
everybody,B
everybody!BeveringBeventie,BeventeenBevent?Bevent-ticketingBevenlyB	even-yearBeven-tarBeven-figureBevelynBevelBevatopolB
evaporatedB	evaporateBevangelicalBevan,B
evaluated.B
evaluated,BevadedBeurtB	eurozone,Beuropol:Beuropean-baedBeurope:Beurojut.Beuro)Beurex,Beuraia,B	euractiv,B	euphemim.Beu:B	eu-impoedB
ettlement)BetranBetoro.B	etiquetteB	etimationBethic,B	ethfinex,B	etherrockB
ethernity,B
etheridge,Bethereum.orgBethereum-likeBethereum-compatibleBetherealBetherdelta.Beth2Beth/udBeth)Betf:Beterbae,Betc.),Betate-relatedBetatalBetablihment.B
etablihed,Bet).BerwerB	ervicing.BervicedBervice;Berupt.BerteBerrano,BeroBerie:BeriaB	ergeyvichBerectedB
erciccioneB
erc20-baedBerc-1155BeramuBeraed.Beraed,B	eradicateBequoia,Bequivalent,BequipBequilibriumBequiBequetrationB	equation,B	equallingBequal.Beptein.Beptein,BeptBepouingBepoue.BeporttarBepiorBepionage-relatedBepinoza,B	epidemic,BephyroB	epenilla,BepenillaBepehrBeparatitB	eparated,Bepa,Bep>43k,BeomBenvoyBenvironment)BenvioulyBenueBentrutBentropyBentrepreneurhip,BentrapB	entitled,BenthuiaticallyB	enthuiat?B	enthuiam.Bentertainer,BentertainerBenterprie-levelBentailedBentail,B	enrollingBenriqueBenrichment,BenriaBenragedB
enormouly,Benor,BenorBenlitingBenjin.BeningBenigma,BenhrinedBenhancement,B	enhanced,Bengagement.Benforcement-centricB	enforced.Benergy-relatedBenergy-baedBenemie,BenembleBenel,BenecaBene?Bendure,BendpointBendorerBending.BendgameBendemicBended,B	endeavor,Bend-uer.Bencouraged,B	encompaedBencodingBenationalitB	enationalB	enactmentBenacted,Ben.BemulatedBemployment,Bemployer-ponoredB	employer,BemphaticB
emotional.BemnB
eminent.lyBemi-abandonedBemeronBemerge,B	emefiele.BembracementBembrace.BembodiedB	embezzledB
embellihedB
embeddableBembargo,B	emanatingBelyiumBelvi!Beluive.BelopeBelon,BelmaaniBellyBelloffBelliott,BelliBell?Bell)Belizavetovka,Belixxir,BelitzerBeliminated.B	eligible,BelieBelfie,Belf-trading,B
elf-titledB
elf-taughtBelf-reportingBelf-regulatedBelf-regulateBelf-publihedBelf-ownerhipB	elf-ownerBelf-overeignB
elf-miningBelf-governingB
elf-fundedBelf-employedBelf-defene.B
elf-defeneBelf-certifiedBelf,BelevierBeleventhBelevatorB
elementaryB	elementalB	electrum,Belectronically,BelectrBelectiveBelectingBelected,BelbowedBelaineBeladBekliptorBekBeip1559B	eip-3554.Beip-3554B	eip-1559,B
einteiniumB
eindhoven,BeikoBeightie.B
eighth-motB
eight-partBeight-figureBehrlichBehramBegwit2mbBegorBegment.BegifterBeg-friendlyBefinityBeffortleBefficiently.B
effectuateBeffectively.Beffectively,B	effected,Beething.BeethingBeet.BeerilyBeepBeential.Beential,Beeing.B	educator,Bedu,Bedt):B	edmonton,B
editorial.BedibleBedgingBedgarB	ecurrencyBecurity:Becurity-centricBectaticBectBecrowedBecretnftBecortedB	econotimeBeconomy:Beconomically,B
economicalB	economic:Becondly,Becond-leadingBecond-layerBecond-generationBecond-biggetBecond),B
ecommerce,B
ecologicalBecocah,BecocahBecobar.BecobankBeco.BeclairBeck,BecedeBecb.BecapingBecapedBecape.B	ecalated,Bec10Bec-regulatedBebon.BebiBebbingB	ebay-likeBebatiBeayitBeayidBeatern/northBeatern,Beater,Beatbch.Bearched-forBeamlely.BealingBeal,BealBeagerneBeagal.BeaementBea.Be9Be10B	e-voucherB
e-hryvnia.Be-franc.Be-fiatBe-currencieBe-cedi,Be-cahBe-bookBe)BdytrophyBdyonBdymonBdydx.BdxyBdwellingBdwarfedBdvdBduty,Bduped.Bduo,BdunnBdumpterBdulyBdukleyBduke,BdukacopyBdugBdudeB	dubrovky,Bdrw,BdrunkBdrummerBdrove,B	dropping.Bdropout,BdropoutBdrive,BdrinkingBdrink.BdrillBdreingBdredenBdreamerB
dreadfullyBdre.BdreBdrawerB	drawback,Bdratically,Bdraper.Bdraghi,Bdox,B
downtrend.B	downtime.BdownplayingB
downplayedBdown:B
down-trendBdow,BdouglaBdoubtingBdoubtedBdoubled.Bdouble-pendingBdoryBdoppelpaymerBdoomed.Bdoomed,Bdoom)BdonutBdont.Bdonovan,Bdonor,BdonningBdonnerB	dongxing,BdongbuBdonatebutton.cah,B	dominguezB
dominationB
dollarizedBdollar]Bdollar?Bdollar-valueBdoldrumBdojcdBdoing?BdogepperBdogemonB
dogefatherBdog-inpiredBdog,Bdoent.B	docu-erieBdocker,BdocBdo-it-yourelfBdnipropetrovkBdna,BdmitriBdmg,Bdm,Bdlt.Bdlive.Bdlive,BdliveBdkkB
djenerate.BdiyanetBdix,BdivyehBdiviiveBdiviibility,BdividingBdivided,Bdivide.BdivertBdiverified,BdivergeBdivan,BdittoB	ditrutingBditrut.Bditributor,Bditributed.BditreedB	ditraughtB
ditractionBditractB	ditortionBditinguihingB
ditinguiheB	ditinctlyBditillBditched,BditateB
diruptive.Bdirty.BdirkenBdirhamBdiregardingB
directorieBdirectionalBdiproportionatelyBdipped.BdipoalB	dipleaureBdiplay.B	diplacingBdipenerB
dipenarie.BdipelledBdipatchBdiparateB
diparagingBdiorder,Bdiolve.Bdinner.BdiningBdineroBdinerBdinehBdinaBdiminihing.Bdimi,BdimezzoB	dimenion.BdilutedBdillyBdilemma.Bdilemma,BdiintermediateBdihonorableBdihingBdihedBdigycodeBdiguie,BdigracedBdignity,Bdigitalization.Bdigital-onlyB	digipule,B
digifinex,B	digifinexBdigetBdifficultie,Bdifferently,B
different;Bdifference).Bdiet,BdientingBdienter.BdienterB	dieminateBdiehardBdieae.BdidainBdictatedB
dicretion,B	dicretionBdicreditB
dicovered,B
dicouragedBdicord,B	dicontentBdicloed,Bdicloe.B
dicernibleBdicernBdiatrou.Bdiarray.BdiarrayBdiar,B
diapprovedBdiappointmentBdiappointedBdiappearing.B	diappear?B	diappear.Bdiapora,B	dialogue,Bdial-upB
diagreeingBdhvcBdhruvBdhanani,Bdhamodharan,B
dhabi-baedBdgbBdfund,Bdfinity,BdexeBdevoutBdevotee.BdevoteeBdevoidB	deviationBdeveloping.B
developer:Bdeveloper/upporterB
developed,B
devatationB	devalued.Bdevaini.Bdeux!Bdetruction.B	detonatedBdetoken,Bdetination,Bdetermined.Bdetermination,BdeteriorationB	detector,B
detection,B	detectingB	detected.Bdetect,Bdetail).Bderry;BderivingBderivative-baedBderibit.BderailBdepth.B	deprivingBdepreciation.B
depreciateB
deprecatedB
depoitory,B	depoitor,B
depoition.B	depoited.Bdeployment,B	depletionBdeplatformed,B
deplatformBdepieBdeperation,BdependedB
dependableB	depatchedB
departure,Bdepartment:BdepartedBdepair,Bdeny.Bdenver,Bdental,Bdent,B	denouncedBdenotingBdenoteBdenominatingBdenmark.Bdenmark-baedBdenleyBdenkiBdenial,BdenetBdenelyBdenBdemytifyBdemontration.Bdemontrated.B
demonetizeBdemographic,Bdemie,BdemeeterB	demanded.BdeltonBdeloreanB
deliverie,Bdeliver.B	deliting.Bdelight,BdeliciouBdeliberation.BdelfiBdeleted,B
delegationBdelegated-proof-of-takeBdeledernierBdelboBdelayed.Bdelayed,B	delaware,BdejanBdeit.BdeiringB	deirable.Bdeigned,Bdeignation,BdeifyingBdehqanBdegree,Bdeflationary.B	deflationBdeflatedBdefipule.com,Bdefined.Bdefined,Bdeficit,B	defenive.B	defender,Bdefeat.BdefacedBdeert,Bdeeper,BdeepenedBdeepen,Bdeepdotweb.comBdeep.B	deecratedBdeductedBdeduceB
dedicationB
decryptionBdecred,BdecoyB
decouplingBdecoupleB
declaring:B	declared,Bdecimal,Bdeciion-makerBdecentraland.BdecendBdecency.B	deceivingB	dec>135k.Bdebut,B
debugging,Bdebated.B	debatableBdebacle.Bdebacle,BdebacleBdeath:BdearthBdeanonymizingBdeanonymizeBdeandreB
dealerhip.Bdea,Bde-factoBde-dollarizationBde-anonymizationBddf8BddexBdbxBdbmBdazzlingBdaymak,BdaxBdavohBdavivienda,Bdavaaambuu,BdatyariBdata-carrier-ize.Bdarkmarket,BdarklyBdark.Bdark,Bdarji,BdaringBdaredBdaraB
dappradar,Bdao.BdanteBdanielleBdanielaBdance,Bdamon.BdamonBdamnedBdamalaBdamaged,BdamagedBdam.BdaliaBdaliBdaleBdagurBdaemon.Bdaa,Bd55cBd/b/aBd&gB
czech-baedBcz,BcyrillicBcyrilB
cypru-baedBcynicBcyclerBcyberthreatBcyberecurity,Bcybercrime,Bcyber-thieveBcyber-criminalBcyber-attackBcw,Bcvc.BcvaBcuttledB	cutthroatBcutom,BcutodiedBcurry,Bcurriculum.Bcurrent,BcurrenexB	currency]B	currency;B	curiouly,BcuredBcure-allBcuratorBcup?Bcup,BcumanB
culturallyB	cultivateBcullion,Bculkin,BculkinBcuevaBcubeBcubano,Bct,B
cryptoweepBcryptowat.chBcryptovere.Bcryptovantage.comBcryptotech.Bcryptoruble.BcryptoradarBcryptoqueen.Bcryptopunk.Bcryptophyl.com,B
cryptopendB
cryptonizeB
cryptonia,B
cryptonautB
cryptomkt,B	cryptomioBcryptomaticBcryptomaniaBcryptojacking,Bcryptographer.Bcryptograph.BcryptoglobalB
cryptofundBcryptofinance.BcryptofacilB
cryptoeat,Bcryptocurrencyalerting.comBcryptocurrency;Bcryptocurrency-tradingBcryptocurrency-linkedBcryptocurrency-centricBcrypto.com.B	crypto.bgB
crypto-yenBcrypto-weep.Bcrypto-traderBcrypto-tatitBcrypto-tartupBcrypto-related.Bcrypto-predictionBcrypto-paceBcrypto-olutionBcrypto-nativeBcrypto-minerBcrypto-mediaBcrypto-lendingBcrypto-kepticBcrypto-invetingBcrypto-inheritanceBcrypto-indutryBcrypto-financeBcrypto-exchangeBcrypto-enthuiat,Bcrypto-economy,Bcrypto-dipeningBcrypto-derivativeBcrypto-cutodyBcrypto-cryptoBcrypto-companie.Bcrypto-community,B
crypto-atmBcrypto-analyiBcrypto-advocateBcrypto-acceptingBcrypto-a-collateralBcryopreervedB	cryogenicBcryexBcrutinized.B	crunchingBcrunchedBcrunch.B
crumbling.BcrumbledBcruie,BcruelBcrubbingBcrubbedBcrrB
crowdourceBcrowdingB	crowdfireBcrowded.BcrotownB	crotower,Bcrore)BcropchoBcronie.Bcro-platformBcro-examinedBcro-categoryBcro,B	critique.B	critique,Bcriticized,B	critical.BcritianoBcriptwriterBcriptoavila,Bcriptan,BcripinBcripBcriminally,Bcriminalized.Bcriminalized,BcriminalityBcrimea,Bcrie,BcrewedB	cret-iuedBcretBcreptBcreepBcreek,B	creditingBcredibility.Bcred.B	creating,Bcreate.BcreamingBcrc,BcrazierBcrawlingBcratch.BcrappingBcrap,BcrammedBcrambledB	craiglit,Bcraig,Bcrahed.B
crackdown:Bcrack,BcraBcpcBcpaBcp2000BcpB	coworker,BcowBcovetBcovered,Bcover-upBcoutureBcourtlitener.com,BcourtlitenerBcourtingB
courthoue.BcourtedBcourBcoupB
countrymenBcountry:B	counting.BcounterpointBcountermeaure.BcounterintelligenceBcountercultureBcounter-economic.Bcounel,Bcould.Bcould,BcouchedBcotly.Bcotly,BcotchBcotanzo,Bcot:BcorteB	corruptedBcorrepondence.BcorreoB
correctly,B
correctiveB
corrected.BcorpeBcorp-backedBcorona,BcoronaBcornedBcorleyBcore).BcordobaB	corcoran,Bcorb
Ā
Const_5Const*
_output_shapes

:??*
dtype0	*??
value??B??	??"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '       '      !'      "'      #'      $'      %'      &'      ''      ('      )'      *'      +'      ,'      -'      .'      /'      0'      1'      2'      3'      4'      5'      6'      7'      8'      9'      :'      ;'      <'      ='      >'      ?'      @'      A'      B'      C'      D'      E'      F'      G'      H'      I'      J'      K'      L'      M'      N'      O'      P'      Q'      R'      S'      T'      U'      V'      W'      X'      Y'      Z'      ['      \'      ]'      ^'      _'      `'      a'      b'      c'      d'      e'      f'      g'      h'      i'      j'      k'      l'      m'      n'      o'      p'      q'      r'      s'      t'      u'      v'      w'      x'      y'      z'      {'      |'      }'      ~'      '      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'       (      (      (      (      (      (      (      (      (      	(      
(      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (       (      !(      "(      #(      $(      %(      &(      '(      ((      )(      *(      +(      ,(      -(      .(      /(      0(      1(      2(      3(      4(      5(      6(      7(      8(      9(      :(      ;(      <(      =(      >(      ?(      @(      A(      B(      C(      D(      E(      F(      G(      H(      I(      J(      K(      L(      M(      N(      O(      P(      Q(      R(      S(      T(      U(      V(      W(      X(      Y(      Z(      [(      \(      ](      ^(      _(      `(      a(      b(      c(      d(      e(      f(      g(      h(      i(      j(      k(      l(      m(      n(      o(      p(      q(      r(      s(      t(      u(      v(      w(      x(      y(      z(      {(      |(      }(      ~(      (      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(       )      )      )      )      )      )      )      )      )      	)      
)      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )       )      !)      ")      #)      $)      %)      &)      ')      ()      ))      *)      +)      ,)      -)      .)      /)      0)      1)      2)      3)      4)      5)      6)      7)      8)      9)      :)      ;)      <)      =)      >)      ?)      @)      A)      B)      C)      D)      E)      F)      G)      H)      I)      J)      K)      L)      M)      N)      O)      P)      Q)      R)      S)      T)      U)      V)      W)      X)      Y)      Z)      [)      \)      ])      ^)      _)      `)      a)      b)      c)      d)      e)      f)      g)      h)      i)      j)      k)      l)      m)      n)      o)      p)      q)      r)      s)      t)      u)      v)      w)      x)      y)      z)      {)      |)      })      ~)      )      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)       *      *      *      *      *      *      *      *      *      	*      
*      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *       *      !*      "*      #*      $*      %*      &*      '*      (*      )*      **      +*      ,*      -*      .*      /*      0*      1*      2*      3*      4*      5*      6*      7*      8*      9*      :*      ;*      <*      =*      >*      ?*      @*      A*      B*      C*      D*      E*      F*      G*      H*      I*      J*      K*      L*      M*      N*      O*      P*      Q*      R*      S*      T*      U*      V*      W*      X*      Y*      Z*      [*      \*      ]*      ^*      _*      `*      a*      b*      c*      d*      e*      f*      g*      h*      i*      j*      k*      l*      m*      n*      o*      p*      q*      r*      s*      t*      u*      v*      w*      x*      y*      z*      {*      |*      }*      ~*      *      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*       +      +      +      +      +      +      +      +      +      	+      
+      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +       +      !+      "+      #+      $+      %+      &+      '+      (+      )+      *+      ++      ,+      -+      .+      /+      0+      1+      2+      3+      4+      5+      6+      7+      8+      9+      :+      ;+      <+      =+      >+      ?+      @+      A+      B+      C+      D+      E+      F+      G+      H+      I+      J+      K+      L+      M+      N+      O+      P+      Q+      R+      S+      T+      U+      V+      W+      X+      Y+      Z+      [+      \+      ]+      ^+      _+      `+      a+      b+      c+      d+      e+      f+      g+      h+      i+      j+      k+      l+      m+      n+      o+      p+      q+      r+      s+      t+      u+      v+      w+      x+      y+      z+      {+      |+      }+      ~+      +      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+       ,      ,      ,      ,      ,      ,      ,      ,      ,      	,      
,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,       ,      !,      ",      #,      $,      %,      &,      ',      (,      ),      *,      +,      ,,      -,      .,      /,      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      :,      ;,      <,      =,      >,      ?,      @,      A,      B,      C,      D,      E,      F,      G,      H,      I,      J,      K,      L,      M,      N,      O,      P,      Q,      R,      S,      T,      U,      V,      W,      X,      Y,      Z,      [,      \,      ],      ^,      _,      `,      a,      b,      c,      d,      e,      f,      g,      h,      i,      j,      k,      l,      m,      n,      o,      p,      q,      r,      s,      t,      u,      v,      w,      x,      y,      z,      {,      |,      },      ~,      ,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,       -      -      -      -      -      -      -      -      -      	-      
-      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -       -      !-      "-      #-      $-      %-      &-      '-      (-      )-      *-      +-      ,-      --      .-      /-      0-      1-      2-      3-      4-      5-      6-      7-      8-      9-      :-      ;-      <-      =-      >-      ?-      @-      A-      B-      C-      D-      E-      F-      G-      H-      I-      J-      K-      L-      M-      N-      O-      P-      Q-      R-      S-      T-      U-      V-      W-      X-      Y-      Z-      [-      \-      ]-      ^-      _-      `-      a-      b-      c-      d-      e-      f-      g-      h-      i-      j-      k-      l-      m-      n-      o-      p-      q-      r-      s-      t-      u-      v-      w-      x-      y-      z-      {-      |-      }-      ~-      -      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-       .      .      .      .      .      .      .      .      .      	.      
.      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .       .      !.      ".      #.      $.      %.      &.      '.      (.      ).      *.      +.      ,.      -.      ..      /.      0.      1.      2.      3.      4.      5.      6.      7.      8.      9.      :.      ;.      <.      =.      >.      ?.      @.      A.      B.      C.      D.      E.      F.      G.      H.      I.      J.      K.      L.      M.      N.      O.      P.      Q.      R.      S.      T.      U.      V.      W.      X.      Y.      Z.      [.      \.      ].      ^.      _.      `.      a.      b.      c.      d.      e.      f.      g.      h.      i.      j.      k.      l.      m.      n.      o.      p.      q.      r.      s.      t.      u.      v.      w.      x.      y.      z.      {.      |.      }.      ~.      .      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.       /      /      /      /      /      /      /      /      /      	/      
/      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /       /      !/      "/      #/      $/      %/      &/      '/      (/      )/      */      +/      ,/      -/      ./      //      0/      1/      2/      3/      4/      5/      6/      7/      8/      9/      :/      ;/      </      =/      >/      ?/      @/      A/      B/      C/      D/      E/      F/      G/      H/      I/      J/      K/      L/      M/      N/      O/      P/      Q/      R/      S/      T/      U/      V/      W/      X/      Y/      Z/      [/      \/      ]/      ^/      _/      `/      a/      b/      c/      d/      e/      f/      g/      h/      i/      j/      k/      l/      m/      n/      o/      p/      q/      r/      s/      t/      u/      v/      w/      x/      y/      z/      {/      |/      }/      ~/      /      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/       0      0      0      0      0      0      0      0      0      	0      
0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0       0      !0      "0      #0      $0      %0      &0      '0      (0      )0      *0      +0      ,0      -0      .0      /0      00      10      20      30      40      50      60      70      80      90      :0      ;0      <0      =0      >0      ?0      @0      A0      B0      C0      D0      E0      F0      G0      H0      I0      J0      K0      L0      M0      N0      O0      P0      Q0      R0      S0      T0      U0      V0      W0      X0      Y0      Z0      [0      \0      ]0      ^0      _0      `0      a0      b0      c0      d0      e0      f0      g0      h0      i0      j0      k0      l0      m0      n0      o0      p0      q0      r0      s0      t0      u0      v0      w0      x0      y0      z0      {0      |0      }0      ~0      0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0       1      1      1      1      1      1      1      1      1      	1      
1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1       1      !1      "1      #1      $1      %1      &1      '1      (1      )1      *1      +1      ,1      -1      .1      /1      01      11      21      31      41      51      61      71      81      91      :1      ;1      <1      =1      >1      ?1      @1      A1      B1      C1      D1      E1      F1      G1      H1      I1      J1      K1      L1      M1      N1      O1      P1      Q1      R1      S1      T1      U1      V1      W1      X1      Y1      Z1      [1      \1      ]1      ^1      _1      `1      a1      b1      c1      d1      e1      f1      g1      h1      i1      j1      k1      l1      m1      n1      o1      p1      q1      r1      s1      t1      u1      v1      w1      x1      y1      z1      {1      |1      }1      ~1      1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1       2      2      2      2      2      2      2      2      2      	2      
2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2       2      !2      "2      #2      $2      %2      &2      '2      (2      )2      *2      +2      ,2      -2      .2      /2      02      12      22      32      42      52      62      72      82      92      :2      ;2      <2      =2      >2      ?2      @2      A2      B2      C2      D2      E2      F2      G2      H2      I2      J2      K2      L2      M2      N2      O2      P2      Q2      R2      S2      T2      U2      V2      W2      X2      Y2      Z2      [2      \2      ]2      ^2      _2      `2      a2      b2      c2      d2      e2      f2      g2      h2      i2      j2      k2      l2      m2      n2      o2      p2      q2      r2      s2      t2      u2      v2      w2      x2      y2      z2      {2      |2      }2      ~2      2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2       3      3      3      3      3      3      3      3      3      	3      
3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3       3      !3      "3      #3      $3      %3      &3      '3      (3      )3      *3      +3      ,3      -3      .3      /3      03      13      23      33      43      53      63      73      83      93      :3      ;3      <3      =3      >3      ?3      @3      A3      B3      C3      D3      E3      F3      G3      H3      I3      J3      K3      L3      M3      N3      O3      P3      Q3      R3      S3      T3      U3      V3      W3      X3      Y3      Z3      [3      \3      ]3      ^3      _3      `3      a3      b3      c3      d3      e3      f3      g3      h3      i3      j3      k3      l3      m3      n3      o3      p3      q3      r3      s3      t3      u3      v3      w3      x3      y3      z3      {3      |3      }3      ~3      3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3       4      4      4      4      4      4      4      4      4      	4      
4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4       4      !4      "4      #4      $4      %4      &4      '4      (4      )4      *4      +4      ,4      -4      .4      /4      04      14      24      34      44      54      64      74      84      94      :4      ;4      <4      =4      >4      ?4      @4      A4      B4      C4      D4      E4      F4      G4      H4      I4      J4      K4      L4      M4      N4      O4      P4      Q4      R4      S4      T4      U4      V4      W4      X4      Y4      Z4      [4      \4      ]4      ^4      _4      `4      a4      b4      c4      d4      e4      f4      g4      h4      i4      j4      k4      l4      m4      n4      o4      p4      q4      r4      s4      t4      u4      v4      w4      x4      y4      z4      {4      |4      }4      ~4      4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4       5      5      5      5      5      5      5      5      5      	5      
5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5       5      !5      "5      #5      $5      %5      &5      '5      (5      )5      *5      +5      ,5      -5      .5      /5      05      15      25      35      45      55      65      75      85      95      :5      ;5      <5      =5      >5      ?5      @5      A5      B5      C5      D5      E5      F5      G5      H5      I5      J5      K5      L5      M5      N5      O5      P5      Q5      R5      S5      T5      U5      V5      W5      X5      Y5      Z5      [5      \5      ]5      ^5      _5      `5      a5      b5      c5      d5      e5      f5      g5      h5      i5      j5      k5      l5      m5      n5      o5      p5      q5      r5      s5      t5      u5      v5      w5      x5      y5      z5      {5      |5      }5      ~5      5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5       6      6      6      6      6      6      6      6      6      	6      
6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6       6      !6      "6      #6      $6      %6      &6      '6      (6      )6      *6      +6      ,6      -6      .6      /6      06      16      26      36      46      56      66      76      86      96      :6      ;6      <6      =6      >6      ?6      @6      A6      B6      C6      D6      E6      F6      G6      H6      I6      J6      K6      L6      M6      N6      O6      P6      Q6      R6      S6      T6      U6      V6      W6      X6      Y6      Z6      [6      \6      ]6      ^6      _6      `6      a6      b6      c6      d6      e6      f6      g6      h6      i6      j6      k6      l6      m6      n6      o6      p6      q6      r6      s6      t6      u6      v6      w6      x6      y6      z6      {6      |6      }6      ~6      6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6       7      7      7      7      7      7      7      7      7      	7      
7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7       7      !7      "7      #7      $7      %7      &7      '7      (7      )7      *7      +7      ,7      -7      .7      /7      07      17      27      37      47      57      67      77      87      97      :7      ;7      <7      =7      >7      ?7      @7      A7      B7      C7      D7      E7      F7      G7      H7      I7      J7      K7      L7      M7      N7      O7      P7      Q7      R7      S7      T7      U7      V7      W7      X7      Y7      Z7      [7      \7      ]7      ^7      _7      `7      a7      b7      c7      d7      e7      f7      g7      h7      i7      j7      k7      l7      m7      n7      o7      p7      q7      r7      s7      t7      u7      v7      w7      x7      y7      z7      {7      |7      }7      ~7      7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7       8      8      8      8      8      8      8      8      8      	8      
8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8       8      !8      "8      #8      $8      %8      &8      '8      (8      )8      *8      +8      ,8      -8      .8      /8      08      18      28      38      48      58      68      78      88      98      :8      ;8      <8      =8      >8      ?8      @8      A8      B8      C8      D8      E8      F8      G8      H8      I8      J8      K8      L8      M8      N8      O8      P8      Q8      R8      S8      T8      U8      V8      W8      X8      Y8      Z8      [8      \8      ]8      ^8      _8      `8      a8      b8      c8      d8      e8      f8      g8      h8      i8      j8      k8      l8      m8      n8      o8      p8      q8      r8      s8      t8      u8      v8      w8      x8      y8      z8      {8      |8      }8      ~8      8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8       9      9      9      9      9      9      9      9      9      	9      
9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9       9      !9      "9      #9      $9      %9      &9      '9      (9      )9      *9      +9      ,9      -9      .9      /9      09      19      29      39      49      59      69      79      89      99      :9      ;9      <9      =9      >9      ?9      @9      A9      B9      C9      D9      E9      F9      G9      H9      I9      J9      K9      L9      M9      N9      O9      P9      Q9      R9      S9      T9      U9      V9      W9      X9      Y9      Z9      [9      \9      ]9      ^9      _9      `9      a9      b9      c9      d9      e9      f9      g9      h9      i9      j9      k9      l9      m9      n9      o9      p9      q9      r9      s9      t9      u9      v9      w9      x9      y9      z9      {9      |9      }9      ~9      9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9       :      :      :      :      :      :      :      :      :      	:      
:      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :       :      !:      ":      #:      $:      %:      &:      ':      (:      ):      *:      +:      ,:      -:      .:      /:      0:      1:      2:      3:      4:      5:      6:      7:      8:      9:      ::      ;:      <:      =:      >:      ?:      @:      A:      B:      C:      D:      E:      F:      G:      H:      I:      J:      K:      L:      M:      N:      O:      P:      Q:      R:      S:      T:      U:      V:      W:      X:      Y:      Z:      [:      \:      ]:      ^:      _:      `:      a:      b:      c:      d:      e:      f:      g:      h:      i:      j:      k:      l:      m:      n:      o:      p:      q:      r:      s:      t:      u:      v:      w:      x:      y:      z:      {:      |:      }:      ~:      :      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:       ;      ;      ;      ;      ;      ;      ;      ;      ;      	;      
;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;       ;      !;      ";      #;      $;      %;      &;      ';      (;      );      *;      +;      ,;      -;      .;      /;      0;      1;      2;      3;      4;      5;      6;      7;      8;      9;      :;      ;;      <;      =;      >;      ?;      @;      A;      B;      C;      D;      E;      F;      G;      H;      I;      J;      K;      L;      M;      N;      O;      P;      Q;      R;      S;      T;      U;      V;      W;      X;      Y;      Z;      [;      \;      ];      ^;      _;      `;      a;      b;      c;      d;      e;      f;      g;      h;      i;      j;      k;      l;      m;      n;      o;      p;      q;      r;      s;      t;      u;      v;      w;      x;      y;      z;      {;      |;      };      ~;      ;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;       <      <      <      <      <      <      <      <      <      	<      
<      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <       <      !<      "<      #<      $<      %<      &<      '<      (<      )<      *<      +<      ,<      -<      .<      /<      0<      1<      2<      3<      4<      5<      6<      7<      8<      9<      :<      ;<      <<      =<      ><      ?<      @<      A<      B<      C<      D<      E<      F<      G<      H<      I<      J<      K<      L<      M<      N<      O<      P<      Q<      R<      S<      T<      U<      V<      W<      X<      Y<      Z<      [<      \<      ]<      ^<      _<      `<      a<      b<      c<      d<      e<      f<      g<      h<      i<      j<      k<      l<      m<      n<      o<      p<      q<      r<      s<      t<      u<      v<      w<      x<      y<      z<      {<      |<      }<      ~<      <      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<       =      =      =      =      =      =      =      =      =      	=      
=      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =       =      !=      "=      #=      $=      %=      &=      '=      (=      )=      *=      +=      ,=      -=      .=      /=      0=      1=      2=      3=      4=      5=      6=      7=      8=      9=      :=      ;=      <=      ==      >=      ?=      @=      A=      B=      C=      D=      E=      F=      G=      H=      I=      J=      K=      L=      M=      N=      O=      P=      Q=      R=      S=      T=      U=      V=      W=      X=      Y=      Z=      [=      \=      ]=      ^=      _=      `=      a=      b=      c=      d=      e=      f=      g=      h=      i=      j=      k=      l=      m=      n=      o=      p=      q=      r=      s=      t=      u=      v=      w=      x=      y=      z=      {=      |=      }=      ~=      =      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=       >      >      >      >      >      >      >      >      >      	>      
>      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >       >      !>      ">      #>      $>      %>      &>      '>      (>      )>      *>      +>      ,>      ->      .>      />      0>      1>      2>      3>      4>      5>      6>      7>      8>      9>      :>      ;>      <>      =>      >>      ?>      @>      A>      B>      C>      D>      E>      F>      G>      H>      I>      J>      K>      L>      M>      N>      O>      P>      Q>      R>      S>      T>      U>      V>      W>      X>      Y>      Z>      [>      \>      ]>      ^>      _>      `>      a>      b>      c>      d>      e>      f>      g>      h>      i>      j>      k>      l>      m>      n>      o>      p>      q>      r>      s>      t>      u>      v>      w>      x>      y>      z>      {>      |>      }>      ~>      >      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>       ?      ?      ?      ?      ?      ?      ?      ?      ?      	?      
?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       ?      !?      "?      #?      $?      %?      &?      '?      (?      )?      *?      +?      ,?      -?      .?      /?      0?      1?      2?      3?      4?      5?      6?      7?      8?      9?      :?      ;?      <?      =?      >?      ??      @?      A?      B?      C?      D?      E?      F?      G?      H?      I?      J?      K?      L?      M?      N?      O?      P?      Q?      R?      S?      T?      U?      V?      W?      X?      Y?      Z?      [?      \?      ]?      ^?      _?      `?      a?      b?      c?      d?      e?      f?      g?      h?      i?      j?      k?      l?      m?      n?      o?      p?      q?      r?      s?      t?      u?      v?      w?      x?      y?      z?      {?      |?      }?      ~?      ?      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??       @      @      @      @      @      @      @      @      @      	@      
@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      !@      "@      #@      $@      %@      &@      '@      (@      )@      *@      +@      ,@      -@      .@      /@      0@      1@      2@      3@      4@      5@      6@      7@      8@      9@      :@      ;@      <@      =@      >@      ?@      @@      A@      B@      C@      D@      E@      F@      G@      H@      I@      J@      K@      L@      M@      N@      O@      P@      Q@      R@      S@      T@      U@      V@      W@      X@      Y@      Z@      [@      \@      ]@      ^@      _@      `@      a@      b@      c@      d@      e@      f@      g@      h@      i@      j@      k@      l@      m@      n@      o@      p@      q@      r@      s@      t@      u@      v@      w@      x@      y@      z@      {@      |@      }@      ~@      @      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@       A      A      A      A      A      A      A      A      A      	A      
A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A       A      !A      "A      #A      $A      %A      &A      'A      (A      )A      *A      +A      ,A      -A      .A      /A      0A      1A      2A      3A      4A      5A      6A      7A      8A      9A      :A      ;A      <A      =A      >A      ?A      @A      AA      BA      CA      DA      EA      FA      GA      HA      IA      JA      KA      LA      MA      NA      OA      PA      QA      RA      SA      TA      UA      VA      WA      XA      YA      ZA      [A      \A      ]A      ^A      _A      `A      aA      bA      cA      dA      eA      fA      gA      hA      iA      jA      kA      lA      mA      nA      oA      pA      qA      rA      sA      tA      uA      vA      wA      xA      yA      zA      {A      |A      }A      ~A      A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A       B      B      B      B      B      B      B      B      B      	B      
B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B       B      !B      "B      #B      $B      %B      &B      'B      (B      )B      *B      +B      ,B      -B      .B      /B      0B      1B      2B      3B      4B      5B      6B      7B      8B      9B      :B      ;B      <B      =B      >B      ?B      @B      AB      BB      CB      DB      EB      FB      GB      HB      IB      JB      KB      LB      MB      NB      OB      PB      QB      RB      SB      TB      UB      VB      WB      XB      YB      ZB      [B      \B      ]B      ^B      _B      `B      aB      bB      cB      dB      eB      fB      gB      hB      iB      jB      kB      lB      mB      nB      oB      pB      qB      rB      sB      tB      uB      vB      wB      xB      yB      zB      {B      |B      }B      ~B      B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B       C      C      C      C      C      C      C      C      C      	C      
C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C       C      !C      "C      #C      $C      %C      &C      'C      (C      )C      *C      +C      ,C      -C      .C      /C      0C      1C      2C      3C      4C      5C      6C      7C      8C      9C      :C      ;C      <C      =C      >C      ?C      @C      AC      BC      CC      DC      EC      FC      GC      HC      IC      JC      KC      LC      MC      NC      OC      PC      QC      RC      SC      TC      UC      VC      WC      XC      YC      ZC      [C      \C      ]C      ^C      _C      `C      aC      bC      cC      dC      eC      fC      gC      hC      iC      jC      kC      lC      mC      nC      oC      pC      qC      rC      sC      tC      uC      vC      wC      xC      yC      zC      {C      |C      }C      ~C      C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C       D      D      D      D      D      D      D      D      D      	D      
D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D       D      !D      "D      #D      $D      %D      &D      'D      (D      )D      *D      +D      ,D      -D      .D      /D      0D      1D      2D      3D      4D      5D      6D      7D      8D      9D      :D      ;D      <D      =D      >D      ?D      @D      AD      BD      CD      DD      ED      FD      GD      HD      ID      JD      KD      LD      MD      ND      OD      PD      QD      RD      SD      TD      UD      VD      WD      XD      YD      ZD      [D      \D      ]D      ^D      _D      `D      aD      bD      cD      dD      eD      fD      gD      hD      iD      jD      kD      lD      mD      nD      oD      pD      qD      rD      sD      tD      uD      vD      wD      xD      yD      zD      {D      |D      }D      ~D      D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D       E      E      E      E      E      E      E      E      E      	E      
E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E       E      !E      "E      #E      $E      %E      &E      'E      (E      )E      *E      +E      ,E      -E      .E      /E      0E      1E      2E      3E      4E      5E      6E      7E      8E      9E      :E      ;E      <E      =E      >E      ?E      @E      AE      BE      CE      DE      EE      FE      GE      HE      IE      JE      KE      LE      ME      NE      OE      PE      QE      RE      SE      TE      UE      VE      WE      XE      YE      ZE      [E      \E      ]E      ^E      _E      `E      aE      bE      cE      dE      eE      fE      gE      hE      iE      jE      kE      lE      mE      nE      oE      pE      qE      rE      sE      tE      uE      vE      wE      xE      yE      zE      {E      |E      }E      ~E      E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E       F      F      F      F      F      F      F      F      F      	F      
F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F       F      !F      "F      #F      $F      %F      &F      'F      (F      )F      *F      +F      ,F      -F      .F      /F      0F      1F      2F      3F      4F      5F      6F      7F      8F      9F      :F      ;F      <F      =F      >F      ?F      @F      AF      BF      CF      DF      EF      FF      GF      HF      IF      JF      KF      LF      MF      NF      OF      PF      QF      RF      SF      TF      UF      VF      WF      XF      YF      ZF      [F      \F      ]F      ^F      _F      `F      aF      bF      cF      dF      eF      fF      gF      hF      iF      jF      kF      lF      mF      nF      oF      pF      qF      rF      sF      tF      uF      vF      wF      xF      yF      zF      {F      |F      }F      ~F      F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F       G      G      G      G      G      G      G      G      G      	G      
G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G       G      !G      "G      #G      $G      %G      &G      'G      (G      )G      *G      +G      ,G      -G      .G      /G      0G      1G      2G      3G      4G      5G      6G      7G      8G      9G      :G      ;G      <G      =G      >G      ?G      @G      AG      BG      CG      DG      EG      FG      GG      HG      IG      JG      KG      LG      MG      NG      OG      PG      QG      RG      SG      TG      UG      VG      WG      XG      YG      ZG      [G      \G      ]G      ^G      _G      `G      aG      bG      cG      dG      eG      fG      gG      hG      iG      jG      kG      lG      mG      nG      oG      pG      qG      rG      sG      tG      uG      vG      wG      xG      yG      zG      {G      |G      }G      ~G      G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G       H      H      H      H      H      H      H      H      H      	H      
H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H       H      !H      "H      #H      $H      %H      &H      'H      (H      )H      *H      +H      ,H      -H      .H      /H      0H      1H      2H      3H      4H      5H      6H      7H      8H      9H      :H      ;H      <H      =H      >H      ?H      @H      AH      BH      CH      DH      EH      FH      GH      HH      IH      JH      KH      LH      MH      NH      OH      PH      QH      RH      SH      TH      UH      VH      WH      XH      YH      ZH      [H      \H      ]H      ^H      _H      `H      aH      bH      cH      dH      eH      fH      gH      hH      iH      jH      kH      lH      mH      nH      oH      pH      qH      rH      sH      tH      uH      vH      wH      xH      yH      zH      {H      |H      }H      ~H      H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H       I      I      I      I      I      I      I      I      I      	I      
I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I       I      !I      "I      #I      $I      %I      &I      'I      (I      )I      *I      +I      ,I      -I      .I      /I      0I      1I      2I      3I      4I      5I      6I      7I      8I      9I      :I      ;I      <I      =I      >I      ?I      @I      AI      BI      CI      DI      EI      FI      GI      HI      II      JI      KI      LI      MI      NI      OI      PI      QI      RI      SI      TI      UI      VI      WI      XI      YI      ZI      [I      \I      ]I      ^I      _I      `I      aI      bI      cI      dI      eI      fI      gI      hI      iI      jI      kI      lI      mI      nI      oI      pI      qI      rI      sI      tI      uI      vI      wI      xI      yI      zI      {I      |I      }I      ~I      I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I       J      J      J      J      J      J      J      J      J      	J      
J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J       J      !J      "J      #J      $J      %J      &J      'J      (J      )J      *J      +J      ,J      -J      .J      /J      0J      1J      2J      3J      4J      5J      6J      7J      8J      9J      :J      ;J      <J      =J      >J      ?J      @J      AJ      BJ      CJ      DJ      EJ      FJ      GJ      HJ      IJ      JJ      KJ      LJ      MJ      NJ      OJ      PJ      QJ      RJ      SJ      TJ      UJ      VJ      WJ      XJ      YJ      ZJ      [J      \J      ]J      ^J      _J      `J      aJ      bJ      cJ      dJ      eJ      fJ      gJ      hJ      iJ      jJ      kJ      lJ      mJ      nJ      oJ      pJ      qJ      rJ      sJ      tJ      uJ      vJ      wJ      xJ      yJ      zJ      {J      |J      }J      ~J      J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J       K      K      K      K      K      K      K      K      K      	K      
K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K       K      !K      "K      #K      $K      %K      &K      'K      (K      )K      *K      +K      ,K      -K      .K      /K      0K      1K      2K      3K      4K      5K      6K      7K      8K      9K      :K      ;K      <K      =K      >K      ?K      @K      AK      BK      CK      DK      EK      FK      GK      HK      IK      JK      KK      LK      MK      NK      OK      PK      QK      RK      SK      TK      UK      VK      WK      XK      YK      ZK      [K      \K      ]K      ^K      _K      `K      aK      bK      cK      dK      eK      fK      gK      hK      iK      jK      kK      lK      mK      nK      oK      pK      qK      rK      sK      tK      uK      vK      wK      xK      yK      zK      {K      |K      }K      ~K      K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K       L      L      L      L      L      L      L      L      L      	L      
L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L       L      !L      "L      #L      $L      %L      &L      'L      (L      )L      *L      +L      ,L      -L      .L      /L      0L      1L      2L      3L      4L      5L      6L      7L      8L      9L      :L      ;L      <L      =L      >L      ?L      @L      AL      BL      CL      DL      EL      FL      GL      HL      IL      JL      KL      LL      ML      NL      OL      PL      QL      RL      SL      TL      UL      VL      WL      XL      YL      ZL      [L      \L      ]L      ^L      _L      `L      aL      bL      cL      dL      eL      fL      gL      hL      iL      jL      kL      lL      mL      nL      oL      pL      qL      rL      sL      tL      uL      vL      wL      xL      yL      zL      {L      |L      }L      ~L      L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L       M      M      M      M      M      M      M      M      M      	M      
M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M       M      !M      "M      #M      $M      %M      &M      'M      (M      )M      *M      +M      ,M      -M      .M      /M      0M      1M      2M      3M      4M      5M      6M      7M      8M      9M      :M      ;M      <M      =M      >M      ?M      @M      AM      BM      CM      DM      EM      FM      GM      HM      IM      JM      KM      LM      MM      NM      OM      PM      QM      RM      SM      TM      UM      VM      WM      XM      YM      ZM      [M      \M      ]M      ^M      _M      `M      aM      bM      cM      dM      eM      fM      gM      hM      iM      jM      kM      lM      mM      nM      oM      pM      qM      rM      sM      tM      uM      vM      wM      xM      yM      zM      {M      |M      }M      ~M      M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M       N      N      N      N      N      N      N      N      N      	N      
N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N       N      !N      "N      #N      $N      %N      &N      'N      (N      )N      *N      +N      ,N      -N      .N      /N      0N      1N      2N      3N      4N      5N      6N      7N      8N      9N      :N      ;N      <N      =N      >N      ?N      @N      AN      BN      CN      DN      EN      FN      GN      HN      IN      JN      KN      LN      MN      NN      ON      PN      QN      RN      SN      TN      UN      VN      WN      XN      YN      ZN      [N      \N      ]N      ^N      _N      `N      aN      bN      cN      dN      eN      fN      gN      hN      iN      jN      kN      lN      mN      nN      oN      pN      qN      rN      sN      tN      uN      vN      wN      xN      yN      zN      {N      |N      }N      ~N      N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N      ?N       O      O      O      O      O      O      O      O      O      	O      
O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O      O       O      !O      "O      #O      $O      %O      &O      'O      (O      )O      *O      +O      ,O      -O      .O      /O      0O      1O      2O      3O      4O      5O      6O      7O      8O      9O      :O      ;O      <O      =O      >O      ?O      @O      AO      BO      CO      DO      EO      FO      GO      HO      IO      JO      KO      LO      MO      NO      OO      PO      QO      RO      SO      TO      UO      VO      WO      XO      YO      ZO      [O      \O      ]O      ^O      _O      `O      aO      bO      cO      dO      eO      fO      gO      hO      iO      jO      kO      lO      mO      nO      oO      pO      qO      rO      sO      tO      uO      vO      wO      xO      yO      zO      {O      |O      }O      ~O      O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O      ?O       P      P      P      P      P      P      P      P      P      	P      
P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P      P       P      !P      "P      #P      $P      %P      &P      'P      (P      )P      *P      +P      ,P      -P      .P      /P      0P      1P      2P      3P      4P      5P      6P      7P      8P      9P      :P      ;P      <P      =P      >P      ?P      @P      AP      BP      CP      DP      EP      FP      GP      HP      IP      JP      KP      LP      MP      NP      OP      PP      QP      RP      SP      TP      UP      VP      WP      XP      YP      ZP      [P      \P      ]P      ^P      _P      `P      aP      bP      cP      dP      eP      fP      gP      hP      iP      jP      kP      lP      mP      nP      oP      pP      qP      rP      sP      tP      uP      vP      wP      xP      yP      zP      {P      |P      }P      ~P      P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P      ?P       Q      Q      Q      Q      Q      Q      Q      Q      Q      	Q      
Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q      Q       Q      !Q      "Q      #Q      $Q      %Q      &Q      'Q      (Q      )Q      *Q      +Q      ,Q      -Q      .Q      /Q      0Q      1Q      2Q      3Q      4Q      5Q      6Q      7Q      8Q      9Q      :Q      ;Q      <Q      =Q      >Q      ?Q      @Q      AQ      BQ      CQ      DQ      EQ      FQ      GQ      HQ      IQ      JQ      KQ      LQ      MQ      NQ      OQ      PQ      QQ      RQ      SQ      TQ      UQ      VQ      WQ      XQ      YQ      ZQ      [Q      \Q      ]Q      ^Q      _Q      `Q      aQ      bQ      cQ      dQ      eQ      fQ      gQ      hQ      iQ      jQ      kQ      lQ      mQ      nQ      oQ      pQ      qQ      rQ      sQ      tQ      uQ      vQ      wQ      xQ      yQ      zQ      {Q      |Q      }Q      ~Q      Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q      ?Q       R      R      R      R      R      R      R      R      R      	R      
R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R      R       R      !R      "R      #R      $R      %R      &R      'R      (R      )R      *R      +R      ,R      -R      .R      /R      0R      1R      2R      3R      4R      5R      6R      7R      8R      9R      :R      ;R      <R      =R      >R      ?R      @R      AR      BR      CR      DR      ER      FR      GR      HR      IR      JR      KR      LR      MR      NR      OR      PR      QR      RR      SR      TR      UR      VR      WR      XR      YR      ZR      [R      \R      ]R      ^R      _R      `R      aR      bR      cR      dR      eR      fR      gR      hR      iR      jR      kR      lR      mR      nR      oR      pR      qR      rR      sR      tR      uR      vR      wR      xR      yR      zR      {R      |R      }R      ~R      R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R      ?R       S      S      S      S      S      S      S      S      S      	S      
S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S      S       S      !S      "S      #S      $S      %S      &S      'S      (S      )S      *S      +S      ,S      -S      .S      /S      0S      1S      2S      3S      4S      5S      6S      7S      8S      9S      :S      ;S      <S      =S      >S      ?S      @S      AS      BS      CS      DS      ES      FS      GS      HS      IS      JS      KS      LS      MS      NS      OS      PS      QS      RS      SS      TS      US      VS      WS      XS      YS      ZS      [S      \S      ]S      ^S      _S      `S      aS      bS      cS      dS      eS      fS      gS      hS      iS      jS      kS      lS      mS      nS      oS      pS      qS      rS      sS      tS      uS      vS      wS      xS      yS      zS      {S      |S      }S      ~S      S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S      ?S       T      T      T      T      T      T      T      T      T      	T      
T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T      T       T      !T      "T      #T      $T      %T      &T      'T      (T      )T      *T      +T      ,T      -T      .T      /T      0T      1T      2T      3T      4T      5T      6T      7T      8T      9T      :T      ;T      <T      =T      >T      ?T      @T      AT      BT      CT      DT      ET      FT      GT      HT      IT      JT      KT      LT      MT      NT      OT      PT      QT      RT      ST      TT      UT      VT      WT      XT      YT      ZT      [T      \T      ]T      ^T      _T      `T      aT      bT      cT      dT      eT      fT      gT      hT      iT      jT      kT      lT      mT      nT      oT      pT      qT      rT      sT      tT      uT      vT      wT      xT      yT      zT      {T      |T      }T      ~T      T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T      ?T       U      U      U      U      U      U      U      U      U      	U      
U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U      U       U      !U      "U      #U      $U      %U      &U      'U      (U      )U      *U      +U      ,U      -U      .U      /U      0U      1U      2U      3U      4U      5U      6U      7U      8U      9U      :U      ;U      <U      =U      >U      ?U      @U      AU      BU      CU      DU      EU      FU      GU      HU      IU      JU      KU      LU      MU      NU      OU      PU      QU      RU      SU      TU      UU      VU      WU      XU      YU      ZU      [U      \U      ]U      ^U      _U      `U      aU      bU      cU      dU      eU      fU      gU      hU      iU      jU      kU      lU      mU      nU      oU      pU      qU      rU      sU      tU      uU      vU      wU      xU      yU      zU      {U      |U      }U      ~U      U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U      ?U       V      V      V      V      V      V      V      V      V      	V      
V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V      V       V      !V      "V      #V      $V      %V      &V      'V      (V      )V      *V      +V      ,V      -V      .V      /V      0V      1V      2V      3V      4V      5V      6V      7V      8V      9V      :V      ;V      <V      =V      >V      ?V      @V      AV      BV      CV      DV      EV      FV      GV      HV      IV      JV      KV      LV      MV      NV      OV      PV      QV      RV      SV      TV      UV      VV      WV      XV      YV      ZV      [V      \V      ]V      ^V      _V      `V      aV      bV      cV      dV      eV      fV      gV      hV      iV      jV      kV      lV      mV      nV      oV      pV      qV      rV      sV      tV      uV      vV      wV      xV      yV      zV      {V      |V      }V      ~V      V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V      ?V       W      W      W      W      W      W      W      W      W      	W      
W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W      W       W      !W      "W      #W      $W      %W      &W      'W      (W      )W      *W      +W      ,W      -W      .W      /W      0W      1W      2W      3W      4W      5W      6W      7W      8W      9W      :W      ;W      <W      =W      >W      ?W      @W      AW      BW      CW      DW      EW      FW      GW      HW      IW      JW      KW      LW      MW      NW      OW      PW      QW      RW      SW      TW      UW      VW      WW      XW      YW      ZW      [W      \W      ]W      ^W      _W      `W      aW      bW      cW      dW      eW      fW      gW      hW      iW      jW      kW      lW      mW      nW      oW      pW      qW      rW      sW      tW      uW      vW      wW      xW      yW      zW      {W      |W      }W      ~W      W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W      ?W       X      X      X      X      X      X      X      X      X      	X      
X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X      X       X      !X      "X      #X      $X      %X      &X      'X      (X      )X      *X      +X      ,X      -X      .X      /X      0X      1X      2X      3X      4X      5X      6X      7X      8X      9X      :X      ;X      <X      =X      >X      ?X      @X      AX      BX      CX      DX      EX      FX      GX      HX      IX      JX      KX      LX      MX      NX      OX      PX      QX      RX      SX      TX      UX      VX      WX      XX      YX      ZX      [X      \X      ]X      ^X      _X      `X      aX      bX      cX      dX      eX      fX      gX      hX      iX      jX      kX      lX      mX      nX      oX      pX      qX      rX      sX      tX      uX      vX      wX      xX      yX      zX      {X      |X      }X      ~X      X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X      ?X       Y      Y      Y      Y      Y      Y      Y      Y      Y      	Y      
Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y      Y       Y      !Y      "Y      #Y      $Y      %Y      &Y      'Y      (Y      )Y      *Y      +Y      ,Y      -Y      .Y      /Y      0Y      1Y      2Y      3Y      4Y      5Y      6Y      7Y      8Y      9Y      :Y      ;Y      <Y      =Y      >Y      ?Y      @Y      AY      BY      CY      DY      EY      FY      GY      HY      IY      JY      KY      LY      MY      NY      OY      PY      QY      RY      SY      TY      UY      VY      WY      XY      YY      ZY      [Y      \Y      ]Y      ^Y      _Y      `Y      aY      bY      cY      dY      eY      fY      gY      hY      iY      jY      kY      lY      mY      nY      oY      pY      qY      rY      sY      tY      uY      vY      wY      xY      yY      zY      {Y      |Y      }Y      ~Y      Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y      ?Y       Z      Z      Z      Z      Z      Z      Z      Z      Z      	Z      
Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z      Z       Z      !Z      "Z      #Z      $Z      %Z      &Z      'Z      (Z      )Z      *Z      +Z      ,Z      -Z      .Z      /Z      0Z      1Z      2Z      3Z      4Z      5Z      6Z      7Z      8Z      9Z      :Z      ;Z      <Z      =Z      >Z      ?Z      @Z      AZ      BZ      CZ      DZ      EZ      FZ      GZ      HZ      IZ      JZ      KZ      LZ      MZ      NZ      OZ      PZ      QZ      RZ      SZ      TZ      UZ      VZ      WZ      XZ      YZ      ZZ      [Z      \Z      ]Z      ^Z      _Z      `Z      aZ      bZ      cZ      dZ      eZ      fZ      gZ      hZ      iZ      jZ      kZ      lZ      mZ      nZ      oZ      pZ      qZ      rZ      sZ      tZ      uZ      vZ      wZ      xZ      yZ      zZ      {Z      |Z      }Z      ~Z      Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z      ?Z       [      [      [      [      [      [      [      [      [      	[      
[      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [      [       [      ![      "[      #[      $[      %[      &[      '[      ([      )[      *[      +[      ,[      -[      .[      /[      0[      1[      2[      3[      4[      5[      6[      7[      8[      9[      :[      ;[      <[      =[      >[      ?[      @[      A[      B[      C[      D[      E[      F[      G[      H[      I[      J[      K[      L[      M[      N[      O[      P[      Q[      R[      S[      T[      U[      V[      W[      X[      Y[      Z[      [[      \[      ][      ^[      _[      `[      a[      b[      c[      d[      e[      f[      g[      h[      i[      j[      k[      l[      m[      n[      o[      p[      q[      r[      s[      t[      u[      v[      w[      x[      y[      z[      {[      |[      }[      ~[      [      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[      ?[       \      \      \      \      \      \      \      \      \      	\      
\      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \      \       \      !\      "\      #\      $\      %\      &\      '\      (\      )\      *\      +\      ,\      -\      .\      /\      0\      1\      2\      3\      4\      5\      6\      7\      8\      9\      :\      ;\      <\      =\      >\      ?\      @\      A\      B\      C\      D\      E\      F\      G\      H\      I\      J\      K\      L\      M\      N\      O\      P\      Q\      R\      S\      T\      U\      V\      W\      X\      Y\      Z\      [\      \\      ]\      ^\      _\      `\      a\      b\      c\      d\      e\      f\      g\      h\      i\      j\      k\      l\      m\      n\      o\      p\      q\      r\      s\      t\      u\      v\      w\      x\      y\      z\      {\      |\      }\      ~\      \      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\      ?\       ]      ]      ]      ]      ]      ]      ]      ]      ]      	]      
]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]      ]       ]      !]      "]      #]      $]      %]      &]      ']      (]      )]      *]      +]      ,]      -]      .]      /]      0]      1]      2]      3]      4]      5]      6]      7]      8]      9]      :]      ;]      <]      =]      >]      ?]      @]      A]      B]      C]      D]      E]      F]      G]      H]      I]      J]      K]      L]      M]      N]      O]      P]      Q]      R]      S]      T]      U]      V]      W]      X]      Y]      Z]      []      \]      ]]      ^]      _]      `]      a]      b]      c]      d]      e]      f]      g]      h]      i]      j]      k]      l]      m]      n]      o]      p]      q]      r]      s]      t]      u]      v]      w]      x]      y]      z]      {]      |]      }]      ~]      ]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]      ?]       ^      ^      ^      ^      ^      ^      ^      ^      ^      	^      
^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^      ^       ^      !^      "^      #^      $^      %^      &^      '^      (^      )^      *^      +^      ,^      -^      .^      /^      0^      1^      2^      3^      4^      5^      6^      7^      8^      9^      :^      ;^      <^      =^      >^      ?^      @^      A^      B^      C^      D^      E^      F^      G^      H^      I^      J^      K^      L^      M^      N^      O^      P^      Q^      R^      S^      T^      U^      V^      W^      X^      Y^      Z^      [^      \^      ]^      ^^      _^      `^      a^      b^      c^      d^      e^      f^      g^      h^      i^      j^      k^      l^      m^      n^      o^      p^      q^      r^      s^      t^      u^      v^      w^      x^      y^      z^      {^      |^      }^      ~^      ^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^      ?^       _      _      _      _      _      _      _      _      _      	_      
_      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _      _       _      !_      "_      #_      $_      %_      &_      '_      (_      )_      *_      +_      ,_      -_      ._      /_      0_      1_      2_      3_      4_      5_      6_      7_      8_      9_      :_      ;_      <_      =_      >_      ?_      @_      A_      B_      C_      D_      E_      F_      G_      H_      I_      J_      K_      L_      M_      N_      O_      P_      Q_      R_      S_      T_      U_      V_      W_      X_      Y_      Z_      [_      \_      ]_      ^_      __      `_      a_      b_      c_      d_      e_      f_      g_      h_      i_      j_      k_      l_      m_      n_      o_      p_      q_      r_      s_      t_      u_      v_      w_      x_      y_      z_      {_      |_      }_      ~_      _      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_      ?_       `      `      `      `      `      `      `      `      `      	`      
`      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `      `       `      !`      "`      #`      $`      %`      &`      '`      (`      )`      *`      +`      ,`      -`      .`      /`      0`      1`      2`      3`      4`      5`      6`      7`      8`      9`      :`      ;`      <`      =`      >`      ?`      @`      A`      B`      C`      D`      E`      F`      G`      H`      I`      J`      K`      L`      M`      N`      O`      P`      Q`      R`      S`      T`      U`      V`      W`      X`      Y`      Z`      [`      \`      ]`      ^`      _`      ``      a`      b`      c`      d`      e`      f`      g`      h`      i`      j`      k`      l`      m`      n`      o`      p`      q`      r`      s`      t`      u`      v`      w`      x`      y`      z`      {`      |`      }`      ~`      `      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`      ?`       a      a      a      a      a      a      a      a      a      	a      
a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a      a       a      !a      "a      #a      $a      %a      &a      'a      (a      )a      *a      +a      ,a      -a      .a      /a      0a      1a      2a      3a      4a      5a      6a      7a      8a      9a      :a      ;a      <a      =a      >a      ?a      @a      Aa      Ba      Ca      Da      Ea      Fa      Ga      Ha      Ia      Ja      Ka      La      Ma      Na      Oa      Pa      Qa      Ra      Sa      Ta      Ua      Va      Wa      Xa      Ya      Za      [a      \a      ]a      ^a      _a      `a      aa      ba      ca      da      ea      fa      ga      ha      ia      ja      ka      la      ma      na      oa      pa      qa      ra      sa      ta      ua      va      wa      xa      ya      za      {a      |a      }a      ~a      a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a      ?a       b      b      b      b      b      b      b      b      b      	b      
b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b      b       b      !b      "b      #b      $b      %b      &b      'b      (b      )b      *b      +b      ,b      -b      .b      /b      0b      1b      2b      3b      4b      5b      6b      7b      8b      9b      :b      ;b      <b      =b      >b      ?b      @b      Ab      Bb      Cb      Db      Eb      Fb      Gb      Hb      Ib      Jb      Kb      Lb      Mb      Nb      Ob      Pb      Qb      Rb      Sb      Tb      Ub      Vb      Wb      Xb      Yb      Zb      [b      \b      ]b      ^b      _b      `b      ab      bb      cb      db      eb      fb      gb      hb      ib      jb      kb      lb      mb      nb      ob      pb      qb      rb      sb      tb      ub      vb      wb      xb      yb      zb      {b      |b      }b      ~b      b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b      ?b       c      c      c      c      c      c      c      c      c      	c      
c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c      c       c      !c      "c      #c      $c      %c      &c      'c      (c      )c      *c      +c      ,c      -c      .c      /c      0c      1c      2c      3c      4c      5c      6c      7c      8c      9c      :c      ;c      <c      =c      >c      ?c      @c      Ac      Bc      Cc      Dc      Ec      Fc      Gc      Hc      Ic      Jc      Kc      Lc      Mc      Nc      Oc      Pc      Qc      Rc      Sc      Tc      Uc      Vc      Wc      Xc      Yc      Zc      [c      \c      ]c      ^c      _c      `c      ac      bc      cc      dc      ec      fc      gc      hc      ic      jc      kc      lc      mc      nc      oc      pc      qc      rc      sc      tc      uc      vc      wc      xc      yc      zc      {c      |c      }c      ~c      c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c      ?c       d      d      d      d      d      d      d      d      d      	d      
d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d      d       d      !d      "d      #d      $d      %d      &d      'd      (d      )d      *d      +d      ,d      -d      .d      /d      0d      1d      2d      3d      4d      5d      6d      7d      8d      9d      :d      ;d      <d      =d      >d      ?d      @d      Ad      Bd      Cd      Dd      Ed      Fd      Gd      Hd      Id      Jd      Kd      Ld      Md      Nd      Od      Pd      Qd      Rd      Sd      Td      Ud      Vd      Wd      Xd      Yd      Zd      [d      \d      ]d      ^d      _d      `d      ad      bd      cd      dd      ed      fd      gd      hd      id      jd      kd      ld      md      nd      od      pd      qd      rd      sd      td      ud      vd      wd      xd      yd      zd      {d      |d      }d      ~d      d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d      ?d       e      e      e      e      e      e      e      e      e      	e      
e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e      e       e      !e      "e      #e      $e      %e      &e      'e      (e      )e      *e      +e      ,e      -e      .e      /e      0e      1e      2e      3e      4e      5e      6e      7e      8e      9e      :e      ;e      <e      =e      >e      ?e      @e      Ae      Be      Ce      De      Ee      Fe      Ge      He      Ie      Je      Ke      Le      Me      Ne      Oe      Pe      Qe      Re      Se      Te      Ue      Ve      We      Xe      Ye      Ze      [e      \e      ]e      ^e      _e      `e      ae      be      ce      de      ee      fe      ge      he      ie      je      ke      le      me      ne      oe      pe      qe      re      se      te      ue      ve      we      xe      ye      ze      {e      |e      }e      ~e      e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e      ?e       f      f      f      f      f      f      f      f      f      	f      
f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f      f       f      !f      "f      #f      $f      %f      &f      'f      (f      )f      *f      +f      ,f      -f      .f      /f      0f      1f      2f      3f      4f      5f      6f      7f      8f      9f      :f      ;f      <f      =f      >f      ?f      @f      Af      Bf      Cf      Df      Ef      Ff      Gf      Hf      If      Jf      Kf      Lf      Mf      Nf      Of      Pf      Qf      Rf      Sf      Tf      Uf      Vf      Wf      Xf      Yf      Zf      [f      \f      ]f      ^f      _f      `f      af      bf      cf      df      ef      ff      gf      hf      if      jf      kf      lf      mf      nf      of      pf      qf      rf      sf      tf      uf      vf      wf      xf      yf      zf      {f      |f      }f      ~f      f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f      ?f       g      g      g      g      g      g      g      g      g      	g      
g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g      g       g      !g      "g      #g      $g      %g      &g      'g      (g      )g      *g      +g      ,g      -g      .g      /g      0g      1g      2g      3g      4g      5g      6g      7g      8g      9g      :g      ;g      <g      =g      >g      ?g      @g      Ag      Bg      Cg      Dg      Eg      Fg      Gg      Hg      Ig      Jg      Kg      Lg      Mg      Ng      Og      Pg      Qg      Rg      Sg      Tg      Ug      Vg      Wg      Xg      Yg      Zg      [g      \g      ]g      ^g      _g      `g      ag      bg      cg      dg      eg      fg      gg      hg      ig      jg      kg      lg      mg      ng      og      pg      qg      rg      sg      tg      ug      vg      wg      xg      yg      zg      {g      |g      }g      ~g      g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g      ?g       h      h      h      h      h      h      h      h      h      	h      
h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h      h       h      !h      "h      #h      $h      %h      &h      'h      (h      )h      *h      +h      ,h      -h      .h      /h      0h      1h      2h      3h      4h      5h      6h      7h      8h      9h      :h      ;h      <h      =h      >h      ?h      @h      Ah      Bh      Ch      Dh      Eh      Fh      Gh      Hh      Ih      Jh      Kh      Lh      Mh      Nh      Oh      Ph      Qh      Rh      Sh      Th      Uh      Vh      Wh      Xh      Yh      Zh      [h      \h      ]h      ^h      _h      `h      ah      bh      ch      dh      eh      fh      gh      hh      ih      jh      kh      lh      mh      nh      oh      ph      qh      rh      sh      th      uh      vh      wh      xh      yh      zh      {h      |h      }h      ~h      h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h      ?h       i      i      i      i      i      i      i      i      i      	i      
i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i      i       i      !i      "i      #i      $i      %i      &i      'i      (i      )i      *i      +i      ,i      -i      .i      /i      0i      1i      2i      3i      4i      5i      6i      7i      8i      9i      :i      ;i      <i      =i      >i      ?i      @i      Ai      Bi      Ci      Di      Ei      Fi      Gi      Hi      Ii      Ji      Ki      Li      Mi      Ni      Oi      Pi      Qi      Ri      Si      Ti      Ui      Vi      Wi      Xi      Yi      Zi      [i      \i      ]i      ^i      _i      `i      ai      bi      ci      di      ei      fi      gi      hi      ii      ji      ki      li      mi      ni      oi      pi      qi      ri      si      ti      ui      vi      wi      xi      yi      zi      {i      |i      }i      ~i      i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i      ?i       j      j      j      j      j      j      j      j      j      	j      
j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j      j       j      !j      "j      #j      $j      %j      &j      'j      (j      )j      *j      +j      ,j      -j      .j      /j      0j      1j      2j      3j      4j      5j      6j      7j      8j      9j      :j      ;j      <j      =j      >j      ?j      @j      Aj      Bj      Cj      Dj      Ej      Fj      Gj      Hj      Ij      Jj      Kj      Lj      Mj      Nj      Oj      Pj      Qj      Rj      Sj      Tj      Uj      Vj      Wj      Xj      Yj      Zj      [j      \j      ]j      ^j      _j      `j      aj      bj      cj      dj      ej      fj      gj      hj      ij      jj      kj      lj      mj      nj      oj      pj      qj      rj      sj      tj      uj      vj      wj      xj      yj      zj      {j      |j      }j      ~j      j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j      ?j       k      k      k      k      k      k      k      k      k      	k      
k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k      k       k      !k      "k      #k      $k      %k      &k      'k      (k      )k      *k      +k      ,k      -k      .k      /k      0k      1k      2k      3k      4k      5k      6k      7k      8k      9k      :k      ;k      <k      =k      >k      ?k      @k      Ak      Bk      Ck      Dk      Ek      Fk      Gk      Hk      Ik      Jk      Kk      Lk      Mk      Nk      Ok      Pk      Qk      Rk      Sk      Tk      Uk      Vk      Wk      Xk      Yk      Zk      [k      \k      ]k      ^k      _k      `k      ak      bk      ck      dk      ek      fk      gk      hk      ik      jk      kk      lk      mk      nk      ok      pk      qk      rk      sk      tk      uk      vk      wk      xk      yk      zk      {k      |k      }k      ~k      k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k      ?k       l      l      l      l      l      l      l      l      l      	l      
l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l      l       l      !l      "l      #l      $l      %l      &l      'l      (l      )l      *l      +l      ,l      -l      .l      /l      0l      1l      2l      3l      4l      5l      6l      7l      8l      9l      :l      ;l      <l      =l      >l      ?l      @l      Al      Bl      Cl      Dl      El      Fl      Gl      Hl      Il      Jl      Kl      Ll      Ml      Nl      Ol      Pl      Ql      Rl      Sl      Tl      Ul      Vl      Wl      Xl      Yl      Zl      [l      \l      ]l      ^l      _l      `l      al      bl      cl      dl      el      fl      gl      hl      il      jl      kl      ll      ml      nl      ol      pl      ql      rl      sl      tl      ul      vl      wl      xl      yl      zl      {l      |l      }l      ~l      l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l      ?l       m      m      m      m      m      m      m      m      m      	m      
m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m      m       m      !m      "m      #m      $m      %m      &m      'm      (m      )m      *m      +m      ,m      -m      .m      /m      0m      1m      2m      3m      4m      5m      6m      7m      8m      9m      :m      ;m      <m      =m      >m      ?m      @m      Am      Bm      Cm      Dm      Em      Fm      Gm      Hm      Im      Jm      Km      Lm      Mm      Nm      Om      Pm      Qm      Rm      Sm      Tm      Um      Vm      Wm      Xm      Ym      Zm      [m      \m      ]m      ^m      _m      `m      am      bm      cm      dm      em      fm      gm      hm      im      jm      km      lm      mm      nm      om      pm      qm      rm      sm      tm      um      vm      wm      xm      ym      zm      {m      |m      }m      ~m      m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m      ?m       n      n      n      n      n      n      n      n      n      	n      
n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n      n       n      !n      "n      #n      $n      %n      &n      'n      (n      )n      *n      +n      ,n      -n      .n      /n      0n      1n      2n      3n      4n      5n      6n      7n      8n      9n      :n      ;n      <n      =n      >n      ?n      @n      An      Bn      Cn      Dn      En      Fn      Gn      Hn      In      Jn      Kn      Ln      Mn      Nn      On      Pn      Qn      Rn      Sn      Tn      Un      Vn      Wn      Xn      Yn      Zn      [n      \n      ]n      ^n      _n      `n      an      bn      cn      dn      en      fn      gn      hn      in      jn      kn      ln      mn      nn      on      pn      qn      rn      sn      tn      un      vn      wn      xn      yn      zn      {n      |n      }n      ~n      n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n      ?n       o      o      o      o      o      o      o      o      o      	o      
o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o       o      !o      "o      #o      $o      %o      &o      'o      (o      )o      *o      +o      ,o      -o      .o      /o      0o      1o      2o      3o      4o      5o      6o      7o      8o      9o      :o      ;o      <o      =o      >o      ?o      @o      Ao      Bo      Co      Do      Eo      Fo      Go      Ho      Io      Jo      Ko      Lo      Mo      No      Oo      Po      Qo      Ro      So      To      Uo      Vo      Wo      Xo      Yo      Zo      [o      \o      ]o      ^o      _o      `o      ao      bo      co      do      eo      fo      go      ho      io      jo      ko      lo      mo      no      oo      po      qo      ro      so      to      uo      vo      wo      xo      yo      zo      {o      |o      }o      ~o      o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o      ?o       p      p      p      p      p      p      p      p      p      	p      
p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p      p       p      !p      "p      #p      $p      %p      &p      'p      (p      )p      *p      +p      ,p      -p      .p      /p      0p      1p      2p      3p      4p      5p      6p      7p      8p      9p      :p      ;p      <p      =p      >p      ?p      @p      Ap      Bp      Cp      Dp      Ep      Fp      Gp      Hp      Ip      Jp      Kp      Lp      Mp      Np      Op      Pp      Qp      Rp      Sp      Tp      Up      Vp      Wp      Xp      Yp      Zp      [p      \p      ]p      ^p      _p      `p      ap      bp      cp      dp      ep      fp      gp      hp      ip      jp      kp      lp      mp      np      op      pp      qp      rp      sp      tp      up      vp      wp      xp      yp      zp      {p      |p      }p      ~p      p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p      ?p       q      q      q      q      q      q      q      q      q      	q      
q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q      q       q      !q      "q      #q      $q      %q      &q      'q      (q      )q      *q      +q      ,q      -q      .q      /q      0q      1q      2q      3q      4q      5q      6q      7q      8q      9q      :q      ;q      <q      =q      >q      ?q      @q      Aq      Bq      Cq      Dq      Eq      Fq      Gq      Hq      Iq      Jq      Kq      Lq      Mq      Nq      Oq      Pq      Qq      Rq      Sq      Tq      Uq      Vq      Wq      Xq      Yq      Zq      [q      \q      ]q      ^q      _q      `q      aq      bq      cq      dq      eq      fq      gq      hq      iq      jq      kq      lq      mq      nq      oq      pq      qq      rq      sq      tq      uq      vq      wq      xq      yq      zq      {q      |q      }q      ~q      q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q      ?q       r      r      r      r      r      r      r      r      r      	r      
r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r      r       r      !r      "r      #r      $r      %r      &r      'r      (r      )r      *r      +r      ,r      -r      .r      /r      0r      1r      2r      3r      4r      5r      6r      7r      8r      9r      :r      ;r      <r      =r      >r      ?r      @r      Ar      Br      Cr      Dr      Er      Fr      Gr      Hr      Ir      Jr      Kr      Lr      Mr      Nr      Or      Pr      Qr      Rr      Sr      Tr      Ur      Vr      Wr      Xr      Yr      Zr      [r      \r      ]r      ^r      _r      `r      ar      br      cr      dr      er      fr      gr      hr      ir      jr      kr      lr      mr      nr      or      pr      qr      rr      sr      tr      ur      vr      wr      xr      yr      zr      {r      |r      }r      ~r      r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r      ?r       s      s      s      s      s      s      s      s      s      	s      
s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s      s       s      !s      "s      #s      $s      %s      &s      's      (s      )s      *s      +s      ,s      -s      .s      /s      0s      1s      2s      3s      4s      5s      6s      7s      8s      9s      :s      ;s      <s      =s      >s      ?s      @s      As      Bs      Cs      Ds      Es      Fs      Gs      Hs      Is      Js      Ks      Ls      Ms      Ns      Os      Ps      Qs      Rs      Ss      Ts      Us      Vs      Ws      Xs      Ys      Zs      [s      \s      ]s      ^s      _s      `s      as      bs      cs      ds      es      fs      gs      hs      is      js      ks      ls      ms      ns      os      ps      qs      rs      ss      ts      us      vs      ws      xs      ys      zs      {s      |s      }s      ~s      s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s      ?s       t      t      t      t      t      t      t      t      t      	t      
t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t      t       t      !t      "t      #t      $t      %t      &t      't      (t      )t      *t      +t      ,t      -t      .t      /t      0t      1t      2t      3t      4t      5t      6t      7t      8t      9t      :t      ;t      <t      =t      >t      ?t      @t      At      Bt      Ct      Dt      Et      Ft      Gt      Ht      It      Jt      Kt      Lt      Mt      Nt      Ot      Pt      Qt      Rt      St      Tt      Ut      Vt      Wt      Xt      Yt      Zt      [t      \t      ]t      ^t      _t      `t      at      bt      ct      dt      et      ft      gt      ht      it      jt      kt      lt      mt      nt      ot      pt      qt      rt      st      tt      ut      vt      wt      xt      yt      zt      {t      |t      }t      ~t      t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t      ?t       u      u      u      u      u      u      u      u      u      	u      
u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u      u       u      !u      "u      #u      $u      %u      &u      'u      (u      )u      *u      +u      ,u      -u      .u      /u      0u      1u      2u      3u      4u      5u      6u      7u      8u      9u      :u      ;u      <u      =u      >u      ?u      @u      Au      Bu      Cu      Du      Eu      Fu      Gu      Hu      Iu      Ju      Ku      Lu      Mu      Nu      Ou      Pu      Qu      Ru      Su      Tu      Uu      Vu      Wu      Xu      Yu      Zu      [u      \u      ]u      ^u      _u      `u      au      bu      cu      du      eu      fu      gu      hu      iu      ju      ku      lu      mu      nu      ou      pu      qu      ru      su      tu      uu      vu      wu      xu      yu      zu      {u      |u      }u      ~u      u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u      ?u       v      v      v      v      v      v      v      v      v      	v      
v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v      v       v      !v      "v      #v      $v      %v      &v      'v      (v      )v      *v      +v      ,v      -v      .v      /v      0v      1v      2v      3v      4v      5v      6v      7v      8v      9v      :v      ;v      <v      =v      >v      ?v      @v      Av      Bv      Cv      Dv      Ev      Fv      Gv      Hv      Iv      Jv      Kv      Lv      Mv      Nv      Ov      Pv      Qv      Rv      Sv      Tv      Uv      Vv      Wv      Xv      Yv      Zv      [v      \v      ]v      ^v      _v      `v      av      bv      cv      dv      ev      fv      gv      hv      iv      jv      kv      lv      mv      nv      ov      pv      qv      rv      sv      tv      uv      vv      wv      xv      yv      zv      {v      |v      }v      ~v      v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v      ?v       w      w      w      w      w      w      w      w      w      	w      
w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w      w       w      !w      "w      #w      $w      %w      &w      'w      (w      )w      *w      +w      ,w      -w      .w      /w      0w      1w      2w      3w      4w      5w      6w      7w      8w      9w      :w      ;w      <w      =w      >w      ?w      @w      Aw      Bw      Cw      Dw      Ew      Fw      Gw      Hw      Iw      Jw      Kw      Lw      Mw      Nw      Ow      Pw      Qw      Rw      Sw      Tw      Uw      Vw      Ww      Xw      Yw      Zw      [w      \w      ]w      ^w      _w      `w      aw      bw      cw      dw      ew      fw      gw      hw      iw      jw      kw      lw      mw      nw      ow      pw      qw      rw      sw      tw      uw      vw      ww      xw      yw      zw      {w      |w      }w      ~w      w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w      ?w       x      x      x      x      x      x      x      x      x      	x      
x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x      x       x      !x      "x      #x      $x      %x      &x      'x      (x      )x      *x      +x      ,x      -x      .x      /x      0x      1x      2x      3x      4x      5x      6x      7x      8x      9x      :x      ;x      <x      =x      >x      ?x      @x      Ax      Bx      Cx      Dx      Ex      Fx      Gx      Hx      Ix      Jx      Kx      Lx      Mx      Nx      Ox      Px      Qx      Rx      Sx      Tx      Ux      Vx      Wx      Xx      Yx      Zx      [x      \x      ]x      ^x      _x      `x      ax      bx      cx      dx      ex      fx      gx      hx      ix      jx      kx      lx      mx      nx      ox      px      qx      rx      sx      tx      ux      vx      wx      xx      yx      zx      {x      |x      }x      ~x      x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x      ?x       y      y      y      y      y      y      y      y      y      	y      
y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y      y       y      !y      "y      #y      $y      %y      &y      'y      (y      )y      *y      +y      ,y      -y      .y      /y      0y      1y      2y      3y      4y      5y      6y      7y      8y      9y      :y      ;y      <y      =y      >y      ?y      @y      Ay      By      Cy      Dy      Ey      Fy      Gy      Hy      Iy      Jy      Ky      Ly      My      Ny      Oy      Py      Qy      Ry      Sy      Ty      Uy      Vy      Wy      Xy      Yy      Zy      [y      \y      ]y      ^y      _y      `y      ay      by      cy      dy      ey      fy      gy      hy      iy      jy      ky      ly      my      ny      oy      py      qy      ry      sy      ty      uy      vy      wy      xy      yy      zy      {y      |y      }y      ~y      y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y      ?y       z      z      z      z      z      z      z      z      z      	z      
z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z      z       z      !z      "z      #z      $z      %z      &z      'z      (z      )z      *z      +z      ,z      -z      .z      /z      0z      1z      2z      3z      4z      5z      6z      7z      8z      9z      :z      ;z      <z      =z      >z      ?z      @z      Az      Bz      Cz      Dz      Ez      Fz      Gz      Hz      Iz      Jz      Kz      Lz      Mz      Nz      Oz      Pz      Qz      Rz      Sz      Tz      Uz      Vz      Wz      Xz      Yz      Zz      [z      \z      ]z      ^z      _z      `z      az      bz      cz      dz      ez      fz      gz      hz      iz      jz      kz      lz      mz      nz      oz      pz      qz      rz      sz      tz      uz      vz      wz      xz      yz      zz      {z      |z      }z      ~z      z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z      ?z       {      {      {      {      {      {      {      {      {      	{      
{      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {      {       {      !{      "{      #{      ${      %{      &{      '{      ({      ){      *{      +{      ,{      -{      .{      /{      0{      1{      2{      3{      4{      5{      6{      7{      8{      9{      :{      ;{      <{      ={      >{      ?{      @{      A{      B{      C{      D{      E{      F{      G{      H{      I{      J{      K{      L{      M{      N{      O{      P{      Q{      R{      S{      T{      U{      V{      W{      X{      Y{      Z{      [{      \{      ]{      ^{      _{      `{      a{      b{      c{      d{      e{      f{      g{      h{      i{      j{      k{      l{      m{      n{      o{      p{      q{      r{      s{      t{      u{      v{      w{      x{      y{      z{      {{      |{      }{      ~{      {      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{      ?{       |      |      |      |      |      |      |      |      |      	|      
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |       |      !|      "|      #|      $|      %|      &|      '|      (|      )|      *|      +|      ,|      -|      .|      /|      0|      1|      2|      3|      4|      5|      6|      7|      8|      9|      :|      ;|      <|      =|      >|      ?|      @|      A|      B|      C|      D|      E|      F|      G|      H|      I|      J|      K|      L|      M|      N|      O|      P|      Q|      R|      S|      T|      U|      V|      W|      X|      Y|      Z|      [|      \|      ]|      ^|      _|      `|      a|      b|      c|      d|      e|      f|      g|      h|      i|      j|      k|      l|      m|      n|      o|      p|      q|      r|      s|      t|      u|      v|      w|      x|      y|      z|      {|      ||      }|      ~|      |      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|      ?|       }      }      }      }      }      }      }      }      }      	}      
}      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }      }       }      !}      "}      #}      $}      %}      &}      '}      (}      )}      *}      +}      ,}      -}      .}      /}      0}      1}      2}      3}      4}      5}      6}      7}      8}      9}      :}      ;}      <}      =}      >}      ?}      @}      A}      B}      C}      D}      E}      F}      G}      H}      I}      J}      K}      L}      M}      N}      O}      P}      Q}      R}      S}      T}      U}      V}      W}      X}      Y}      Z}      [}      \}      ]}      ^}      _}      `}      a}      b}      c}      d}      e}      f}      g}      h}      i}      j}      k}      l}      m}      n}      o}      p}      q}      r}      s}      t}      u}      v}      w}      x}      y}      z}      {}      |}      }}      ~}      }      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}      ?}       ~      ~      ~      ~      ~      ~      ~      ~      ~      	~      
~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~      ~       ~      !~      "~      #~      $~      %~      &~      '~      (~      )~      *~      +~      ,~      -~      .~      /~      0~      1~      2~      3~      4~      5~      6~      7~      8~      9~      :~      ;~      <~      =~      >~      ?~      @~      A~      B~      C~      D~      E~      F~      G~      H~      I~      J~      K~      L~      M~      N~      O~      P~      Q~      R~      S~      T~      U~      V~      W~      X~      Y~      Z~      [~      \~      ]~      ^~      _~      `~      a~      b~      c~      d~      e~      f~      g~      h~      i~      j~      k~      l~      m~      n~      o~      p~      q~      r~      s~      t~      u~      v~      w~      x~      y~      z~      {~      |~      }~      ~~      ~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~      ?~                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_92136
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_92141
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?1
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?0
value?0B?0 B?0
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
"
_lookup_layer
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratemfmgmh#mi$mj)mk*mlvmvnvo#vp$vq)vr*vs
1
1
2
3
#4
$5
)6
*7
1
0
1
2
#3
$4
)5
*6
 
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
		variables

trainable_variables
regularization_losses
 
3
9lookup_table
:token_counts
;	keras_api
 
db
VARIABLE_VALUEEmbedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
 trainable_variables
!regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6

Z0
[1
 
 

\_initializer
LJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	]total
	^count
_	variables
`	keras_api
D
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

_	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

d	variables
??
VARIABLE_VALUEAdam/Embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConstConst_1Const_2Embedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_91670
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(Embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/Embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp/Adam/Embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst_6*-
Tin&
$2"		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_92268
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameEmbedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotalcounttotal_1count_1Adam/Embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/Embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_92371??

?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91975

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?n
?
E__inference_sequential_layer_call_and_return_conditional_losses_91564
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
embedding_91543:
??
dense_91547:
dense_91549:
dense_1_91553:
dense_1_91555:
dense_2_91558:
dense_2_91560:
identity??!Embedding/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!Embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_91543*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_91181?
(global_average_pooling1d/PartitionedCallPartitionedCall*Embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91190?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_91547dense_91549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_91203?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_91214?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_91553dense_1_91555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_91227?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_91558dense_2_91560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_91243w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^Embedding/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2F
!Embedding/StatefulPartitionedCall!Embedding/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
T
8__inference_global_average_pooling1d_layer_call_fn_91963

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91190`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
F
__inference__creator_92084
identity: ??MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?o
?
E__inference_sequential_layer_call_and_return_conditional_losses_91635
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
embedding_91614:
??
dense_91618:
dense_91620:
dense_1_91624:
dense_1_91626:
dense_2_91629:
dense_2_91631:
identity??!Embedding/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!Embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_91614*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_91181?
(global_average_pooling1d/PartitionedCallPartitionedCall*Embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91190?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_91618dense_91620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_91203?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_91315?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_91624dense_1_91626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_91227?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_91629dense_2_91631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_91243w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^Embedding/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2F
!Embedding/StatefulPartitionedCall!Embedding/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91115

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?	
 __inference__wrapped_model_91105
text_vectorization_inputZ
Vsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle[
Wsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	7
3sequential_text_vectorization_string_lookup_equal_y:
6sequential_text_vectorization_string_lookup_selectv2_t	?
+sequential_embedding_embedding_lookup_91076:
??A
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:C
1sequential_dense_2_matmul_readvariableop_resource:@
2sequential_dense_2_biasadd_readvariableop_resource:
identity??%sequential/Embedding/embedding_lookup?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2w
)sequential/text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
0sequential/text_vectorization/StaticRegexReplaceStaticRegexReplace2sequential/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite p
/sequential/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
7sequential/text_vectorization/StringSplit/StringSplitV2StringSplitV29sequential/text_vectorization/StaticRegexReplace:output:08sequential/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
=sequential/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
?sequential/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
?sequential/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
7sequential/text_vectorization/StringSplit/strided_sliceStridedSliceAsequential/text_vectorization/StringSplit/StringSplitV2:indices:0Fsequential/text_vectorization/StringSplit/strided_slice/stack:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_1:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
?sequential/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/text_vectorization/StringSplit/strided_slice_1StridedSlice?sequential/text_vectorization/StringSplit/StringSplitV2:shape:0Hsequential/text_vectorization/StringSplit/strided_slice_1/stack:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
`sequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast@sequential/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastBsequential/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapedsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
nsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterrsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0wsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastpsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2qsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulmsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
msequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumtsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2tsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Vsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle@sequential/text_vectorization/StringSplit/StringSplitV2:values:0Wsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
1sequential/text_vectorization/string_lookup/EqualEqual@sequential/text_vectorization/StringSplit/StringSplitV2:values:03sequential_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
4sequential/text_vectorization/string_lookup/SelectV2SelectV25sequential/text_vectorization/string_lookup/Equal:z:06sequential_text_vectorization_string_lookup_selectv2_tRsequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
4sequential/text_vectorization/string_lookup/IdentityIdentity=sequential/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????|
:sequential/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
2sequential/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
Asequential/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor;sequential/text_vectorization/RaggedToTensor/Const:output:0=sequential/text_vectorization/string_lookup/Identity:output:0Csequential/text_vectorization/RaggedToTensor/default_value:output:0Bsequential/text_vectorization/StringSplit/strided_slice_1:output:0@sequential/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
%sequential/Embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_91076Jsequential/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*>
_class4
20loc:@sequential/Embedding/embedding_lookup/91076*+
_output_shapes
:?????????x*
dtype0?
.sequential/Embedding/embedding_lookup/IdentityIdentity.sequential/Embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/Embedding/embedding_lookup/91076*+
_output_shapes
:?????????x?
0sequential/Embedding/embedding_lookup/Identity_1Identity7sequential/Embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????x|
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/global_average_pooling1d/MeanMean9sequential/Embedding/embedding_lookup/Identity_1:output:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/dense/MatMulMatMul1sequential/global_average_pooling1d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????~
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*'
_output_shapes
:??????????
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#sequential/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential/Embedding/embedding_lookup(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOpJ^sequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2N
%sequential/Embedding/embedding_lookup%sequential/Embedding/embedding_lookup2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2?
Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
T
8__inference_global_average_pooling1d_layer_call_fn_91958

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91115i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
:
__inference__creator_92066
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1029*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?

?
@__inference_dense_layer_call_and_return_conditional_losses_91995

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_92010

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_92061

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?o
?
E__inference_sequential_layer_call_and_return_conditional_losses_91441

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
embedding_91420:
??
dense_91424:
dense_91426:
dense_1_91430:
dense_1_91432:
dense_2_91435:
dense_2_91437:
identity??!Embedding/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!Embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_91420*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_91181?
(global_average_pooling1d/PartitionedCallPartitionedCall*Embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91190?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_91424dense_91426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_91203?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_91315?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_91430dense_1_91432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_91227?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_91435dense_2_91437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_91243w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^Embedding/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2F
!Embedding/StatefulPartitionedCall!Embedding/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_92079
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
,
__inference__destroyer_92094
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
D__inference_Embedding_layer_call_and_return_conditional_losses_91181

inputs	*
embedding_lookup_91175:
??
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_91175inputs*
Tindices0	*)
_class
loc:@embedding_lookup/91175*+
_output_shapes
:?????????x*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/91175*+
_output_shapes
:?????????x?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????xw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????xY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_91315

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_91724

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_91441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_save_fn_92113
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
__inference_restore_fn_92121
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?B
?
__inference_adapt_step_91937
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
'__inference_dense_1_layer_call_fn_92031

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_91227o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_92042

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_91493
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_91441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_91227

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?}
?
E__inference_sequential_layer_call_and_return_conditional_losses_91889

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	4
 embedding_embedding_lookup_91853:
??6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??Embedding/embedding_lookup?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Embedding/embedding_lookupResourceGather embedding_embedding_lookup_91853?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@Embedding/embedding_lookup/91853*+
_output_shapes
:?????????x*
dtype0?
#Embedding/embedding_lookup/IdentityIdentity#Embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@Embedding/embedding_lookup/91853*+
_output_shapes
:?????????x?
%Embedding/embedding_lookup/Identity_1Identity,Embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????xq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMean.Embedding/embedding_lookup/Identity_1:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Embedding/embedding_lookup^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 28
Embedding/embedding_lookupEmbedding/embedding_lookup2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_91214

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_92000

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_91214`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
__inference__traced_save_92268
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??::::::: : : : : ::: : : : :
??:::::::
??::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::!

_output_shapes
: 
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91969

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
*
__inference_<lambda>_92141
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_92022

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_91203

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_92051

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_91243o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_91275
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_91250o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_91243

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__initializer_92089
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
#__inference_signature_wrapper_91670
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_91105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
*__inference_sequential_layer_call_fn_91697

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_91250o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_921367
3key_value_init1028_lookuptableimportv2_table_handle/
+key_value_init1028_lookuptableimportv2_keys1
-key_value_init1028_lookuptableimportv2_values	
identity??&key_value_init1028/LookupTableImportV2?
&key_value_init1028/LookupTableImportV2LookupTableImportV23key_value_init1028_lookuptableimportv2_table_handle+key_value_init1028_lookuptableimportv2_keys-key_value_init1028_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1028/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??2P
&key_value_init1028/LookupTableImportV2&key_value_init1028/LookupTableImportV2:"

_output_shapes

:??:"

_output_shapes

:??
?
`
'__inference_dropout_layer_call_fn_92005

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_91315o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?m
?
E__inference_sequential_layer_call_and_return_conditional_losses_91250

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
embedding_91182:
??
dense_91204:
dense_91206:
dense_1_91228:
dense_1_91230:
dense_2_91244:
dense_2_91246:
identity??!Embedding/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!Embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_91182*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_91181?
(global_average_pooling1d/PartitionedCallPartitionedCall*Embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91190?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_91204dense_91206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_91203?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_91214?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_91228dense_1_91230*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_91227?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_91244dense_2_91246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_91243w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^Embedding/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 2F
!Embedding/StatefulPartitionedCall!Embedding/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_Embedding_layer_call_and_return_conditional_losses_91953

inputs	*
embedding_lookup_91947:
??
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_91947inputs*
Tindices0	*)
_class
loc:@embedding_lookup/91947*+
_output_shapes
:?????????x*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/91947*+
_output_shapes
:?????????x?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????xw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????xY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?

)__inference_Embedding_layer_call_fn_91944

inputs	
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????x*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_91181s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????x`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????x: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91190

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????x:S O
+
_output_shapes
:?????????x
 
_user_specified_nameinputs
?~
?
!__inference__traced_restore_92371
file_prefix9
%assignvariableop_embedding_embeddings:
??1
assignvariableop_1_dense_kernel:+
assignvariableop_2_dense_bias:3
!assignvariableop_3_dense_1_kernel:-
assignvariableop_4_dense_1_bias:3
!assignvariableop_5_dense_2_kernel:-
assignvariableop_6_dense_2_bias:&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: #
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: C
/assignvariableop_16_adam_embedding_embeddings_m:
??9
'assignvariableop_17_adam_dense_kernel_m:3
%assignvariableop_18_adam_dense_bias_m:;
)assignvariableop_19_adam_dense_1_kernel_m:5
'assignvariableop_20_adam_dense_1_bias_m:;
)assignvariableop_21_adam_dense_2_kernel_m:5
'assignvariableop_22_adam_dense_2_bias_m:C
/assignvariableop_23_adam_embedding_embeddings_v:
??9
'assignvariableop_24_adam_dense_kernel_v:3
%assignvariableop_25_adam_dense_bias_v:;
)assignvariableop_26_adam_dense_1_kernel_v:5
'assignvariableop_27_adam_dense_1_bias_v:;
)assignvariableop_28_adam_dense_2_kernel_v:5
'assignvariableop_29_adam_dense_2_bias_v:
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:12RestoreV2:tensors:13*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_dense_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_1_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_1_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_2_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_2_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?
?
%__inference_dense_layer_call_fn_91984

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_91203o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_920747
3key_value_init1028_lookuptableimportv2_table_handle/
+key_value_init1028_lookuptableimportv2_keys1
-key_value_init1028_lookuptableimportv2_values	
identity??&key_value_init1028/LookupTableImportV2?
&key_value_init1028/LookupTableImportV2LookupTableImportV23key_value_init1028_lookuptableimportv2_table_handle+key_value_init1028_lookuptableimportv2_keys-key_value_init1028_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1028/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??2P
&key_value_init1028/LookupTableImportV2&key_value_init1028/LookupTableImportV2:"

_output_shapes

:??:"

_output_shapes

:??
?v
?
E__inference_sequential_layer_call_and_return_conditional_losses_91803

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	4
 embedding_embedding_lookup_91774:
??6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??Embedding/embedding_lookup?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern[%s]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????x       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????x*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Embedding/embedding_lookupResourceGather embedding_embedding_lookup_91774?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@Embedding/embedding_lookup/91774*+
_output_shapes
:?????????x*
dtype0?
#Embedding/embedding_lookup/IdentityIdentity#Embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@Embedding/embedding_lookup/91774*+
_output_shapes
:?????????x?
%Embedding/embedding_lookup/Identity_1Identity,Embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????xq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMean.Embedding/embedding_lookup/Identity_1:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Embedding/embedding_lookup^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : : : 28
Embedding/embedding_lookupEmbedding/embedding_lookup2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0?????????=
dense_22
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
t__call__
*u&call_and_return_all_conditional_losses
v_default_save_signature"
_tf_keras_sequential
P
_lookup_layer
	keras_api
w_adapt_function"
_tf_keras_layer
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratemfmgmh#mi$mj)mk*mlvmvnvo#vp$vq)vr*vs"
	optimizer
Q
1
2
3
#4
$5
)6
*7"
trackable_list_wrapper
Q
0
1
2
#3
$4
)5
*6"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
		variables

trainable_variables
regularization_losses
t__call__
v_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
L
9lookup_table
:token_counts
;	keras_api"
_tf_keras_layer
"
_generic_user_object
(:&
??2Embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
 trainable_variables
!regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
m
\_initializer
?_create_resource
?_initialize
?_destroy_resourceR jCustom.StaticHashTable
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	]total
	^count
_	variables
`	keras_api"
_tf_keras_metric
^
	atotal
	bcount
c
_fn_kwargs
d	variables
e	keras_api"
_tf_keras_metric
"
_generic_user_object
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
-:+
??2Adam/Embedding/embeddings/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
-:+
??2Adam/Embedding/embeddings/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
?2?
*__inference_sequential_layer_call_fn_91275
*__inference_sequential_layer_call_fn_91697
*__inference_sequential_layer_call_fn_91724
*__inference_sequential_layer_call_fn_91493?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_91803
E__inference_sequential_layer_call_and_return_conditional_losses_91889
E__inference_sequential_layer_call_and_return_conditional_losses_91564
E__inference_sequential_layer_call_and_return_conditional_losses_91635?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_91105text_vectorization_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_91937?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Embedding_layer_call_fn_91944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Embedding_layer_call_and_return_conditional_losses_91953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_global_average_pooling1d_layer_call_fn_91958
8__inference_global_average_pooling1d_layer_call_fn_91963?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91969
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91975?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_91984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_91995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_92000
'__inference_dropout_layer_call_fn_92005?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_92010
B__inference_dropout_layer_call_and_return_conditional_losses_92022?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_92031?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_92042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_92051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_92061?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_91670text_vectorization_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_92066?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_92074?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_92079?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_92084?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_92089?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_92094?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_92113checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_92121restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5?
D__inference_Embedding_layer_call_and_return_conditional_losses_91953_/?,
%?"
 ?
inputs?????????x	
? ")?&
?
0?????????x
? 
)__inference_Embedding_layer_call_fn_91944R/?,
%?"
 ?
inputs?????????x	
? "??????????x6
__inference__creator_92066?

? 
? "? 6
__inference__creator_92084?

? 
? "? 8
__inference__destroyer_92079?

? 
? "? 8
__inference__destroyer_92094?

? 
? "? A
__inference__initializer_920749???

? 
? "? :
__inference__initializer_92089?

? 
? "? ?
 __inference__wrapped_model_91105?9???#$)*=?:
3?0
.?+
text_vectorization_input?????????
? "1?.
,
dense_2!?
dense_2?????????j
__inference_adapt_step_91937J:???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
B__inference_dense_1_layer_call_and_return_conditional_losses_92042\#$/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_92031O#$/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_92061\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_92051O)*/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_91995\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_91984O/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_92010\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_92022\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? z
'__inference_dropout_layer_call_fn_92000O3?0
)?&
 ?
inputs?????????
p 
? "??????????z
'__inference_dropout_layer_call_fn_92005O3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91969{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_91975`7?4
-?*
$?!
inputs?????????x

 
? "%?"
?
0?????????
? ?
8__inference_global_average_pooling1d_layer_call_fn_91958nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
8__inference_global_average_pooling1d_layer_call_fn_91963S7?4
-?*
$?!
inputs?????????x

 
? "??????????y
__inference_restore_fn_92121Y:K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_92113?:&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
E__inference_sequential_layer_call_and_return_conditional_losses_91564~9???#$)*E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_91635~9???#$)*E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_91803l9???#$)*3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_91889l9???#$)*3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_91275q9???#$)*E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_91493q9???#$)*E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_91697_9???#$)*3?0
)?&
?
inputs?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_91724_9???#$)*3?0
)?&
?
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_91670?9???#$)*Y?V
? 
O?L
J
text_vectorization_input.?+
text_vectorization_input?????????"1?.
,
dense_2!?
dense_2?????????