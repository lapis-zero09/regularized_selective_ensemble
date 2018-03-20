import py4j.GatewayServer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.DenseInstance;
import weka.core.matrix.Matrix;
import weka.classifiers.functions.supportVector.RBFKernel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.lang.StringBuilder;

public class GetKernelLinkMatrix {
    //the number of intances in the data
    protected int m_numInst;
    protected int m_numAttr;

    /**
     * get the matrix from Python
     * @return Instances
     * @throws Exception
    */
    public Instances createFromPy4j(byte[] data) throws Exception {
        ByteBuffer buf = ByteBuffer.wrap(data);
        m_numInst = buf.getInt();
        m_numAttr = buf.getInt();
        FastVector attributes = new FastVector();
        for (int i = 0; i < m_numAttr; ++i)
            attributes.addElement(new Attribute(Integer.toString(i)));

        Instances m_data = new Instances("da", attributes, 0);
        for (int i = 0; i < m_numInst; ++i) {
            Instance instance = new DenseInstance(m_numAttr);
            for (int j = 0; j < m_numAttr; ++j)
                instance.setValue(j, buf.getDouble());
            m_data.add(instance);
        }
        return m_data;
    }

    /**
     * save the matrix W for the File
     * @return String
     * @throws Exception
    */
    public String createFile(Matrix matrix, int fileNum, int start, int end, String fold) throws Exception {
        StringBuilder builder = new StringBuilder();
        for(int i = start; i < end; i++) {
            for(int j = 0; j < m_numInst; j++) {
                  // System.out.println(matrix.get(i, j));
                  builder.append(matrix.get(i, j)+"");
                  if(j < m_numInst - 1)
                      builder.append(",");
            }
            if (i % 100 == 0)
              System.out.println(i);
            builder.append("\n");
        }

        String filename = "../data/" + fold + "/w_link/w_link_" + fileNum + ".csv";
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
        writer.write(builder.toString());
        builder = null;
        writer.close();
        writer = null;
        matrix = null;
        System.out.println(filename);
        return filename;
    }

    /**
     * return file split arr
     * @return int[]
     * @throws Exception
    */
    public int[] makeNumlist(int m_numInst) throws Exception {
        int[] arr = {0, 0, 0, 0, m_numInst};
        int num = (m_numInst / 4);
        for (int i = 1; i < arr.length-1; i++) {
          arr[i] = num * i;
        }
        return arr;
    }

    /**
     * return save file name
     * @return String[]
     * @throws Exception
    */
    public String[] returnFilename(Matrix matrix, int m_numInst, String fold) throws Exception {
        int[] arr = makeNumlist(m_numInst);
        String[] filenames = new String[arr.length-1];
        for (int i = 0; i < arr.length-1; i++) {
            filenames[i] = createFile(matrix, i, arr[i], arr[i+1], fold);
        }
        return filenames;
    }

    /**
     * get the matrix W for Laplacian
     * @return w_link filename
     * @throws Exception
     */
    public String[] getKernelLinkMatrix(byte[] data, String fold) throws Exception {
        System.out.println("\t\t[*] Hello from Java!");
        // get the link matrix
        // create java matrix from numpy
        Instances d = createFromPy4j(data);
        Matrix LMat = new Matrix(m_numInst, m_numInst);
        RBFKernel krl = new RBFKernel();
        double g = 0.5 / m_numAttr;
        krl.setGamma(g);
        krl.buildKernel(d);

        System.out.println("\t\t[*] calculating w_link on Java...");
        for (int i = 0; i < m_numInst; i++) {
            for (int j = 0; j <= i; j++) {
                double vt = krl.eval(i, j, d.instance(i));
                LMat.set(i, j, vt);
                LMat.set(j, i, vt);
            }
        }
        d = null;
        krl = null;
        System.out.println("\t\t[-] calc done...");

        return returnFilename(LMat, m_numInst, fold);
    }

    /**
     * connect to Python
     * @return
     * @throws Exception
    */
    public static void main(String[] args) throws Exception {
        GetKernelLinkMatrix app = new GetKernelLinkMatrix();
        GatewayServer server = new GatewayServer(app);
        server.start();
        System.out.println("  [!] py4j Gateway Server Started");
    }
}
