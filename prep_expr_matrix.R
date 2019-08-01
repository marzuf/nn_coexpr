# Rscript prep_expr_matrix.R
# 
cat("!!! WARNING: everything hard-coded !!!")

script_name <- "prep_expr_matrix.R"

startTime <- Sys.time()

cat("> START ", script_name, "\n")

require(foreach)
require(doMC)
require(reshape2)

SSHFS <- FALSE
setDir <- ifelse(SSHFS, "/media/electron", "")

registerDoMC(ifelse(SSHFS, 2, 40))

outDir <- "INPUT_AGG_40kb"
dir.create(outDir, recursive = TRUE)

binSize <- 40*10^3

corMet <- "spearman"

entrezDT_file <- paste0(setDir, "/mnt/ed4/marie/entrez2synonym/entrez/ENTREZ_POS/gff_entrez_position_GRCh37p13_nodup.txt")
stopifnot(file.exists(entrezDT_file))
entrezDT <- read.delim(entrezDT_file, header=TRUE, stringsAsFactors = FALSE)


exprDT_file <- paste0("GSE71862_MCF7_MCF10A_RSEM_expectedcounts.txt")
stopifnot(file.exists(exprDT_file))
exprDT <- read.delim(exprDT_file, header=TRUE)
exprDT$accession <- NULL

outPrefix <- gsub(".txt", "", basename(exprDT_file))

stopifnot(!duplicated(exprDT$gene))

stopifnot(!is.na(exprDT))

rownames(exprDT) <- exprDT$gene
exprDT$gene <- NULL

cat(paste0("av. gene positions: ", sum(rownames(exprDT) %in% entrezDT$symbol), "/", nrow(exprDT), " (", round(sum(rownames(exprDT) %in% entrezDT$symbol)/nrow(exprDT)*100,2), "%)\n"))

entrezDT <- entrezDT[entrezDT$symbol %in% rownames(exprDT),]

entrezDT$TSS <- ifelse(entrezDT$strand == "+", entrezDT$start,
                       ifelse(entrezDT$strand == "-", entrezDT$end, NA))
stopifnot(!is.na(entrezDT$TSS))                       

all_chr <- unique(entrezDT$chromo)


chromo = all_chr[1]
foo <- foreach(chromo = all_chr) %dopar% {
  
  chr_genes <- entrezDT$symbol[entrezDT$chromo == chromo]
  
  chr_exprDT <- exprDT[chr_genes,]
  
  stopifnot(length(chr_genes) == nrow(chr_exprDT))
  
  chr_corExprMat <- cor(t(chr_exprDT), method=corMet)
  
  stopifnot(dim(chr_corExprMat) == length(chr_genes))
  stopifnot(colnames(chr_corExprMat) == rownames(chr_corExprMat))
  
  chr_corExprMat[lower.tri(chr_corExprMat, diag=TRUE)] <- NA
  
  chr_coexprDT <- melt(chr_corExprMat)
  colnames(chr_coexprDT) <- c("gene1", "gene2", "coexpr")
  
  cat(paste0(chromo, " - # NA: ", sum(is.na(chr_coexprDT)), "/", nrow(chr_coexprDT), " (", round(sum(is.na(chr_coexprDT))/nrow(chr_coexprDT)*100, 2), "%)\n"))
  
  chr_coexprDT <- na.omit(chr_coexprDT)
  
  colnames(chr_coexprDT)[colnames(chr_coexprDT) == "gene1"] <- "symbol"
  chr_coexprDT_withPos1 <- merge(chr_coexprDT, entrezDT[,c("symbol", "TSS")], by="symbol", all.x = TRUE, all.y = FALSE)
  stopifnot(!is.na(chr_coexprDT_withPos1$TSS))
  stopifnot(nrow(chr_coexprDT_withPos1) == nrow(chr_coexprDT))
  colnames(chr_coexprDT_withPos1)[colnames(chr_coexprDT_withPos1) == "symbol"] <- "gene1"
  colnames(chr_coexprDT_withPos1)[colnames(chr_coexprDT_withPos1) == "TSS"] <- "TSS_gene1"
  colnames(chr_coexprDT_withPos1)[colnames(chr_coexprDT_withPos1) == "gene2"] <- "symbol"
  chr_coexprDT_withPos <- merge(chr_coexprDT_withPos1, entrezDT[,c("symbol", "TSS")], by="symbol", all.x = TRUE, all.y = FALSE)
  rm(chr_coexprDT_withPos1)
  stopifnot(!is.na(chr_coexprDT_withPos$TSS))
  colnames(chr_coexprDT_withPos)[colnames(chr_coexprDT_withPos) == "symbol"] <- "gene2"
  colnames(chr_coexprDT_withPos)[colnames(chr_coexprDT_withPos) == "TSS"] <- "TSS_gene2"
  
  chr_coexprDT_withPos$binA_1 <- chr_coexprDT_withPos$TSS_gene1%/%binSize
  chr_coexprDT_withPos$binB_1 <- chr_coexprDT_withPos$TSS_gene2%/%binSize
  
  chr_coexprDT_withPos$binA <- pmin(chr_coexprDT_withPos$binA_1, chr_coexprDT_withPos$binB_1)
  chr_coexprDT_withPos$binB <- pmax(chr_coexprDT_withPos$binA_1, chr_coexprDT_withPos$binB_1)
  
  
  stopifnot(chr_coexprDT_withPos$binA == chr_coexprDT_withPos$binA_1 | chr_coexprDT_withPos$binA == chr_coexprDT_withPos$binB_1)
  stopifnot(chr_coexprDT_withPos$binB == chr_coexprDT_withPos$binA_1 | chr_coexprDT_withPos$binB == chr_coexprDT_withPos$binB_1)
  
  stopifnot(chr_coexprDT_withPos$binA <= chr_coexprDT_withPos$binB)
  
  chr_coexprDT_withPos$binA_1 <- NULL
  chr_coexprDT_withPos$binB_1 <- NULL
  
  
  chr_agg_coexprDT <- aggregate(coexpr ~ binA + binB, data = chr_coexprDT_withPos, FUN=mean)

  stopifnot(is.numeric(chr_agg_coexprDT$binA))
  stopifnot(is.numeric(chr_agg_coexprDT$binB))
  
  chr_agg_coexprDT <- chr_agg_coexprDT[order(chr_agg_coexprDT$binA, chr_agg_coexprDT$binB),]
  
  outFile <- file.path(outDir, paste0(outPrefix, "_", chromo, "_agg.txt"))
  write.table(chr_agg_coexprDT, col.names = FALSE, row.names = FALSE, sep="\t", quote=FALSE, append=FALSE, file = outFile)
  cat(paste0("... written: ", outFile, "\n"))
  
}


########################################################################################################
########################################################################################################
########################################################################################################
cat("*** DONE: ", script_name, "\n")
cat(paste0(startTime, "\n", Sys.time()))


