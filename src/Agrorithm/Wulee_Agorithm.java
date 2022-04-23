/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Agrorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

/**
 *
 * @author hpmdu
 */
public class Wulee_Agorithm {
    
    static{
        // Load thư viện vào
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    
    private Mat coverImage = null;
    private Mat key = null;
    private int blockWidth = 0;
    private int blockHeight = 0;
    private String message = "";
    private String bin_message = "";
    private int hiddenCount = 0;
    private String retrieveMessage = "";
    private String bin_retrieve = "";
    private String bin_retrieve_temp = "";
    private int retrieveCount = 0;
    private int retrieveMax = -1;
    private boolean retrieve_done = false;
    private int CHANNEL_BLUE = 0;
    private int CHANNEL_GREEN = 1;
    private int CHANNEL_RED = 2;
    
    public void setKey(String keyString){
        // Khởi tạo ma trận khóa với kích thước rows = keyString.length(), cols = 8
        key = new Mat(keyString.length(), 8, CvType.CV_8UC1);
        for(int i = 0; i < keyString.length(); i++){
            // binString là chuyển đổi của 1 ký tự trong keyString sang dạng chuỗi nhị phân có độ dài là 8 -> %8s
            String binString = String.format("%8s", Integer.toBinaryString(keyString.charAt(i))).replaceAll(" ", "0");
            // Gán chuỗi nhị phân trên vào đúng vị trí trong ma trận key
            for(int j = 0; j < 8; j++){
                key.row(i).col(j).setTo(new Scalar(Double.parseDouble(String.format("%c", binString.charAt(j)))));
            }
        }
        blockHeight = key.rows();
        blockWidth = key.cols();
        System.out.println(String.format("key: %s\n%s", keyString, key.dump()));
        System.out.println("key_sum: " + (int)Core.sumElems(key).val[0]);
    }
    
    public void setKeyToNull(){
        key = null;
    }
    
    public void setCoverImage(String path){
        // Đọc theo cấu trúc BGR
        coverImage = new Imgcodecs().imread(path);
    }
    
    public boolean isCoverImageNull(){
        return coverImage == null;
    }
    
    public void saveStegoImage(String path){
        Imgcodecs.imwrite(path, coverImage);
    }
    
    public void setMessage(String message){
        this.message = String.format("%d:%s", message.length(), message);
        for(int i = 0; i < this.message.length(); i++){
            this.bin_message += String.format("%8s", Integer.toBinaryString(this.message.charAt(i))).replaceAll(" ", "0");
        }
//        System.out.println("Bin message: " + this.bin_message);
    }

    public String getRetrieveMessage() {
        return retrieveMessage;
    }

//    public void setRetrieveMax(int retrieveMax) {
//        // retrieveMax giờ đây sẽ là số lượng bit tối đa có thể lấy
//        this.retrieveMax = retrieveMax * 8;
//    }
    
    public void resetWhenHide(){
        message = "";
        bin_message = "";
        hiddenCount = 0;
    }
    
    public void resetWhenRetrieve(){
        retrieveCount = 0;
        retrieveMax = -1;
        retrieveMessage = "";
        bin_retrieve = "";
        bin_retrieve_temp = "";
        retrieve_done = false;
    }
    
    public String calculate(){
        int pixelCount = getTotalPixels();
        int keyLengthMax = getCoverImageHeight();
        return String.format("\nTotal number of pixels: %d"
                + "\nKey length maximum: %s"
                + "\nImage height: %d"
                + "\nImage width : %d", pixelCount, keyLengthMax, coverImage.rows(), coverImage.cols());
    }
    
    public int getCoverImageHeight(){
        return coverImage.rows();
    }
    
    public int getCoverImageWidth(){
        return coverImage.cols();
    }
    
    public int getTotalPixels(){
        return coverImage.rows() * coverImage.cols();
    }
    
    public Mat fi_to_binary(Mat col){
        Mat fi = new Mat(blockHeight, blockWidth, CvType.CV_8UC1, new Scalar(0));
        // Duyệt theo chiều cao - theo hàng vì col truyền vô là 1 ma trận (blockHeight,1)
        for(int i = 0; i < blockHeight; i++){
            String binary = String.format("%8s", 
                    Integer.toBinaryString((int)col.get(i, 0)[0])).replaceAll(" ", "0");
            // Mặc định blockWidth = 8
            for(int j = 0; j < blockWidth; j++){
                // Set 1 hoặc 0 theo chuỗi binary ở trên
                fi.row(i).col(j).setTo(new Scalar(binary.charAt(j) == '1'? 1.0 : 0.0));
            }
        }
        
        return fi;
    }
    
    public Mat fi_to_int(Mat fi){
        Mat newFi = new Mat(blockHeight,1,CvType.CV_8UC1, new Scalar(0));
        for(int i = 0; i < blockHeight; i++){
            String binString = "";
            for(int j = 0; j < blockWidth; j++){
                // .get sẽ return array double vì Mat là ma trận 3 chiều
                binString += (int)fi.get(i, j)[0];
            }
            // col(0) là vì newFi chỉ có 1 cột
            newFi.col(0).row(i).setTo(new Scalar((double)Integer.parseInt(binString, 2)));
        }
        return newFi;
    }
    
    public Mat randomBinaryReplace(Mat fi, int target, int replace){
        for(int i = 0; i < fi.rows(); i++){
            for(int j = 0; j < fi.cols(); j++){
                if ((int)fi.get(i, j)[0] == target && (int)key.get(i, j)[0] == 1){
//                    if (target == 1) System.out.println(fi.dump());
                    fi.row(i).col(j).setTo(new Scalar(replace));
//                    if (target == 1) System.out.println(fi.dump());
                    return fi;
                }
            }
        }
        return fi;
    }
    
    public Mat randomBinaryReverse(Mat fi){
        for(int i = 0; i < fi.rows(); i++){
            for(int j = 0; j < fi.cols(); j++){
                if ((int)key.get(i, j)[0] == 1){
                    int rev = (int)fi.get(i, j)[0] == 1 ? 0:1;
                    fi.row(i).col(j).setTo(new Scalar(rev));
                    return fi;
                }
            }
        }
        return fi;
    }
    
    public Mat hideInBlock(Mat block,  char c_bit, int channel, int rowPos, int colPos){ // này là giấu tin vào từng block trong ảnh
        // char c_bit ở trên là bit thông tin cần được giấu
        // block ở trên có 3 channel BGR, (blockHeight,blockWidth,3) BGR
        List<Mat> listChannel = new ArrayList<Mat>();
        // Core.split là phương thức tách chiều sau của ma trận rồi return vào 1 list<Mat> đã khởi tạo ở trên
        Core.split(block, listChannel);
        // channel_block_int là ma trận 2 chiều của 1 channel của block ở trên
        Mat channel_block_int = listChannel.get(channel);
        
        // Fi là ma trận nhị phân
        Mat fi = fi_to_binary(channel_block_int);
        // Function Core.bitwise_and giữa fi và key, sau đó return kết quả vào biến matAnd
        Mat matXor = new Mat();
        Core.bitwise_xor(fi, key, matXor);
        // Function Core.sumElems là phương thức cộng tất cả phần tử trong ma trận lại
        int xor_sum = (int)Core.sumElems(matXor).val[0];
        int key_sum = (int)Core.sumElems(key).val[0];
        // xor_sum > 0 && xor_sum < key_sum
        if (xor_sum > 0 && xor_sum < key_sum){
            System.out.printf("fi_bin_before: \n%s\n", fi.dump());
            int bit = c_bit == '1' ? 1:0;
            if (xor_sum % 2 == bit){
                
            }
            else if (xor_sum == 1){
//                fi = randomBinaryReplace(fi, 0, 1);
                fi = randomBinaryReplace(fi, 1, 0);
            }
            else if (xor_sum == key_sum - 1){
//                fi = randomBinaryReplace(fi, 1, 0);
                fi = randomBinaryReplace(fi, 0, 1);
            }
            else {
                fi = randomBinaryReverse(fi);
            }
            // In ra màn hình giá trị fi binary mới
                System.out.printf("fi_bin_after :\n%s\n",fi.dump());
            Core.bitwise_xor(fi, key, matXor);
            int new_xor_sum = (int)Core.sumElems(matXor).val[0];
            fi = fi_to_int(fi);
            
            // In ra màn hình giá trị fi int mới
//                System.out.printf("fi[%d] int :\n%s\n",i,fi.dump());
            // Gán giá trị fi mới vào channel_block_int.
            channel_block_int = fi;
            hiddenCount+=1;
            System.out.println(String.format("xor_sum: %d | Vị trí đúng: r=%d,c=%d | new_xor_sum: %d | Bit đang giấu: %d", xor_sum, rowPos, colPos, new_xor_sum, bit));
        }
//        else System.out.println(String.format("xor_sum: %d | Vị trí ko đúng: %d", xor_sum, pos));
        
//        // Sau khi channel_block đã được chỉnh sửa, ta gán ngược lại vào listChannel ở trên đúng với thứ tự channel
        listChannel.set(channel, channel_block_int);
        // Sau đó merge listChannel mới vào Mat block để ra 1 block ảnh mới đã được giấu tin và return nó về
        Core.merge(listChannel, block);
//        System.out.println(block.dump());
        // Sau khi dấu xong thì hiddenCount sẽ tăng lên 1 vì đã dấu được 1 ký tự
        return block;
    }
    
    public void assignNewBlockToCoverImage(Mat block, int row, int col){
        // Đây là phương thức gán block(BGR) vào ảnh(BGR) - row, col là điểm bắt đầu trong ma trận
        for(int i = 0; i < block.rows(); i++){
            for(int j = 0; j < block.cols(); j++){
                coverImage.put(row + i, col + j, block.get(i, j));
            }
        }
    }
    
    public void hideInChannel(int channel){
        // Duyệt chiều cao ảnh
        for(int i = 0; i < getCoverImageHeight(); i += blockHeight){
            if (i + blockHeight > getCoverImageHeight() || hiddenCount == bin_message.length()) return;
            for(int j = 0; j < getCoverImageWidth(); j += 1){
                if (hiddenCount == bin_message.length()) return;
                // Trích xuất block - 1 block này sẽ dấu được 1 bit, block này là block int có shape (blockHeight,1)
//                Mat block = coverImage.colRange(j,j + 1).rowRange(i,i + blockHeight);
                Mat block = coverImage.col(j).rowRange(i,i + blockHeight);
                // block mới sau khi được dấu 1 bit
                block = hideInBlock(block, bin_message.charAt(hiddenCount), channel, i, j);
                // gán block mới vào ảnh
                assignNewBlockToCoverImage(block, i, j);
            }
        }
    }
    
    public void hide(){
        System.out.println("\nGiấu tin");
        hideInChannel(CHANNEL_BLUE);
        hideInChannel(CHANNEL_GREEN);
        hideInChannel(CHANNEL_RED);
    }
    
    public void retrieveInBlock(Mat block, int channel, int rowPos, int colPos){
        // char c_bit ở trên là bit thông tin cần được giấu
        // block ở trên có 3 channel BGR, (blockHeight,blockWidth,3) BGR
        List<Mat> listChannel = new ArrayList<Mat>();
        // Core.split là phương thức tách chiều sau của ma trận rồi return vào 1 list<Mat> đã khởi tạo ở trên
        Core.split(block, listChannel);
        // channel_block_int là ma trận 2 chiều của 1 channel của block ở trên
        Mat channel_block_int = listChannel.get(channel);
        
        // Fi là ma trận nhị phân
        Mat fi = fi_to_binary(channel_block_int);
        // Function Core.bitwise_and giữa fi và key, sau đó return kết quả vào biến matAnd
        Mat matXor = new Mat();
        Core.bitwise_xor(fi, key, matXor);
        // Function Core.sumElems là phương thức cộng tất cả phần tử trong ma trận lại
        int xor_sum = (int)Core.sumElems(matXor).val[0];
        int key_sum = (int)Core.sumElems(key).val[0];
        
        // xor_sum > 0 && xor_sum < key_sum
        if (xor_sum > 0 && xor_sum < key_sum){
            if (xor_sum % 2 == 0){
                bin_retrieve_temp += "0";
            }
            else {
                bin_retrieve_temp += "1";
            }
            
            System.out.println(String.format("xor_sum: %d | Vị trí đúng: r=%d,c=%d", xor_sum, rowPos, colPos));
            // bin_retrieve_temp là 1 biến dùng để thu thập 8 bit, nếu khi đã đủ 8 bit thì sẽ được đổi thành 1 ký tự
            if (bin_retrieve_temp.length() == 8){
                // đổi bin_retrieve_temp đã đủ 8 bit thành 1 số ascii
                int ascii = Integer.parseInt(bin_retrieve_temp, 2);
                // Nếu số ascii đó nằm trong khoảng kí tự có ý nghĩa
                // ascii = 10 là new line \n
                if (ascii >= 32 && ascii <= 126 || ascii == 10) {
                    // 58 là kí tự ':', nếu như đụng kí tự ':' thì sẽ parse thành 1 số int 
                    if (ascii == 58 && retrieveMax == -1){
                        try{
                            retrieveMax = Integer.parseInt(retrieveMessage) + 1;
                        }
                        catch(NumberFormatException e){
                            retrieve_done = true;
                        }
                        retrieveMessage = "";
                    }
                    
                    // Sau khi đã trích xuất được độ dài tin nhán8
                    if (retrieveMax > 0) retrieveCount++;
                    
                    retrieveMessage += String.format("%c", (char)ascii);
                    bin_retrieve += bin_retrieve_temp;
                    bin_retrieve_temp = "";
                }
                else {
                    retrieve_done = true;
                    System.out.println("ascii out: " + ascii);
                }
            }
        }
//        else System.out.println(String.format("xor_sum: %d | Vị trí ko đúng: %d", xor_sum, pos));
    }
    
    public void retrieveInChannel(int channel){ // giải mã ảnh từ những block trong channel ảnh .
        if (retrieve_done) return;
        for(int i = 0; i < getCoverImageHeight(); i += blockHeight){
            if (i + blockHeight > getCoverImageHeight() || retrieveCount == retrieveMax) return;
            for(int j = 0; j < getCoverImageWidth(); j += 1){
                if (retrieve_done || retrieveCount == retrieveMax) return;
//                Mat block = coverImage.colRange(j,j + 1).rowRange(i,i + blockHeight);
                Mat block = coverImage.col(j).rowRange(i,i + blockHeight);    
                retrieveInBlock(block, channel, i, j);
            }
        }
    }
    
    public void retrieve(){
        // Kiểm lỗi
        System.out.println("\nTrích xuất");
        retrieveInChannel(CHANNEL_BLUE);
        retrieveInChannel(CHANNEL_GREEN);
        retrieveInChannel(CHANNEL_RED);
        if (retrieveMessage.length() != 0) retrieveMessage = retrieveMessage.substring(1);
    }
    
    public void compareTest(){
//        System.out.println("\nMa trận khóa:\n" + this.key.dump() + "\n");
        System.out.println("\nSo sánh ");
        System.out.println("Raw message     : " + this.message);
        System.out.println("Bin message     : " + this.bin_message);
        System.out.println("Bin retrieve    : " + this.bin_retrieve);
        System.out.println("retrieve message: " + this.retrieveMessage);
    }
    
    public static void main(String[] args) {
        String msg = "a";
        Wulee_Agorithm wulee = new Wulee_Agorithm();
        wulee.setKey("123456");
        wulee.setCoverImage("cat.jpeg");
        wulee.setMessage(msg);
        wulee.hide();
        wulee.retrieve();
        wulee.compareTest();
//        System.out.println(wulee.getRetrieveMessage());
//        System.out.println(Pattern.matches("(.*).png", "ab-c.png"));
    }
}
