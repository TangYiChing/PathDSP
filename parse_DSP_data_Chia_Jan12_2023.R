rm(list = ls())

setwd("c:/Users/m092469/OneDrive - Mayo Clinic/temp_code/DOE_IMPROVE")

options(stringsAsFactors = FALSE)

rDrug <- read.delim(
  "data/GDSCv2.Gao2015.Powell2020.Lee2021.GeoSearch.Ding2016.CHEM.256.MBits.txt")

rGSEA <- read.delim(
  "data/GDSCv2.Powell2020.EXP.ssGSEA.txt")

rPNET <- read.delim(
  "data/GDSCv2.Gao2015.Powell2020.Lee2021.GeoSearch.Ding2016.DGNet.NetPEA.txt")

rReponse0 <- read.delim("data/GDSCv2.resp_PowellAUC.Alias.txt")

Drug.ID.vec0 <- intersect(rDrug$drug, rReponse0$Therapy)
Cell.ID.vec <- intersect(rGSEA$X, rReponse0$Sample)

Drug.ID.vec <- intersect(Drug.ID.vec0,rPNET$X)

sel.Reponse.idx <- which(is.element(rReponse0$Therapy, Drug.ID.vec) &
                           is.element(rReponse0$Sample, Cell.ID.vec))

rReponse <- rReponse0[sel.Reponse.idx, ]

N.cell <- length(Cell.ID.vec)
N.drug <- length(Drug.ID.vec)
N.comb <- nrow(rReponse)

head(rDrug[,1:5])
Drug.fmtx <- data.matrix(rDrug[match(Drug.ID.vec, rDrug$drug), 2: ncol(rDrug)])
rownames(Drug.fmtx) <- Drug.ID.vec
head(Drug.fmtx[,1:5])

Drug.PNEA.fmtx <- data.matrix(rPNET[match(Drug.ID.vec, rPNET$X), 2: ncol(rPNET)])
rownames(Drug.PNEA.fmtx) <- Drug.ID.vec
head(Drug.PNEA.fmtx[,1:5])

all(rownames(Drug.fmtx)==rownames(Drug.PNEA.fmtx))

Cell.GSEA.mtx <- data.matrix(rGSEA[match(Cell.ID.vec, rGSEA$X), 2: ncol(rGSEA)])
rownames(Cell.GSEA.mtx) <- Cell.ID.vec
head(Cell.GSEA.mtx[,1:5])

N.col <- ncol(Drug.fmtx) + ncol(Drug.PNEA.fmtx) + 
  ncol(Cell.GSEA.mtx) + 3 # drug, cell & resp
comb.data.mtx <- mat.or.vec(N.comb, N.col)
colnames(comb.data.mtx) <- c("drug", "cell",
                             paste0("feature",1:(N.col-3)),"resp")

 
for(i in 1:N.comb){
    # i <- 1
    tmp.cell.ID <- rReponse$Sample[i]    
    tmp.drug.ID <- rReponse$Therapy[i]
    comb.data.mtx[i, "drug"] <- tmp.drug.ID
    comb.data.mtx[i, "cell"] <- tmp.cell.ID
    comb.data.mtx[i, paste0("feature",1:(N.col-3))] <- 
      c(Drug.fmtx[tmp.drug.ID,],
        Drug.PNEA.fmtx[tmp.drug.ID,],
        Cell.GSEA.mtx[tmp.cell.ID,])
    response.idx <-which(
      rReponse$Therapy==tmp.drug.ID & rReponse$Sample==tmp.cell.ID)
    tmp.resp <- rReponse$Response[response.idx]
    comb.data.mtx[i, "resp"] <- tmp.resp
   
}

write.table(x = comb.data.mtx, file = "input_txt_Nick.txt", 
            quote = FALSE, sep = "\t", row.names = FALSE)