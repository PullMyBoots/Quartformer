package qfm_fi;

import java.io.BufferedWriter;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;

public class FastMain {

    private static final String DEFAULT_TAXON_PREFIX = "t";

    private static long readLongLE(byte[] buf, int off) {
        long b0 = buf[off] & 0xFFL;
        long b1 = buf[off + 1] & 0xFFL;
        long b2 = buf[off + 2] & 0xFFL;
        long b3 = buf[off + 3] & 0xFFL;
        long b4 = buf[off + 4] & 0xFFL;
        long b5 = buf[off + 5] & 0xFFL;
        long b6 = buf[off + 6] & 0xFFL;
        long b7 = buf[off + 7] & 0xFFL;
        return (b0)
                | (b1 << 8)
                | (b2 << 16)
                | (b3 << 24)
                | (b4 << 32)
                | (b5 << 40)
                | (b6 << 48)
                | (b7 << 56);
    }

    private static Taxa getOrCreateTaxa(Map<String, Taxa> taxMap, String name) {
        Taxa t = taxMap.get(name);
        if (t == null) {
            t = new Taxa(name);
            taxMap.put(name, t);
        }
        return t;
    }

    private static String solveFromQuartetList(ArrayList<Quartet> qr, Map<String, Taxa> taxMap) {
        qr.sort(Comparator.comparing(Quartet::getQFrequency, Collections.reverseOrder()));
        LinkedHashSet<Taxa> taxaList = new LinkedHashSet<Taxa>(taxMap.values());
        taxMap.clear();
        String s = Routines.SQP(qr, taxaList, 1000, 0);
        if (s == null) {
            s = "null";
        }
        return s;
    }

    private static String runBinaryInput(String inputPath, String taxonPrefix) {
        long start = System.currentTimeMillis();
        Map<String, Taxa> taxMap = new HashMap<String, Taxa>();
        ArrayList<Quartet> qr = new ArrayList<Quartet>();
        long records = 0;

        final int recordSize = 16;
        final int chunkBytes = 16 * 1024 * 1024; // 16MB
        byte[] buf = new byte[chunkBytes + recordSize];
        int carry = 0;
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(inputPath), chunkBytes)) {
            while (true) {
                int read = bis.read(buf, carry, chunkBytes);
                if (read < 0) {
                    break;
                }
                int total = carry + read;
                int limit = total - (total % recordSize);
                for (int off = 0; off < limit; off += recordSize) {
                    long key = readLongLE(buf, off);
                    long weightLong = readLongLE(buf, off + 8);

                    if (weightLong <= 0) {
                        continue;
                    }
                    if (weightLong > Integer.MAX_VALUE) {
                        throw new IllegalArgumentException(
                                "quartet weight overflow int32: " + weightLong + " at record " + records);
                    }

                    int a = (int) ((key >>> 48) & 0xFFFFL);
                    int b = (int) ((key >>> 32) & 0xFFFFL);
                    int c = (int) ((key >>> 16) & 0xFFFFL);
                    int d = (int) (key & 0xFFFFL);

                    Taxa t1 = getOrCreateTaxa(taxMap, taxonPrefix + a);
                    Taxa t2 = getOrCreateTaxa(taxMap, taxonPrefix + b);
                    Taxa t3 = getOrCreateTaxa(taxMap, taxonPrefix + c);
                    Taxa t4 = getOrCreateTaxa(taxMap, taxonPrefix + d);
                    qr.add(new Quartet(t1, t2, t3, t4, (int) weightLong));
                    records++;
                }
                carry = total - limit;
                if (carry > 0) {
                    System.arraycopy(buf, limit, buf, 0, carry);
                }
            }
            if (carry != 0) {
                throw new IllegalArgumentException(
                        "binary quartet file has trailing bytes not aligned to 16-byte records: " + carry);
            }
        } catch (Exception e) {
            throw new RuntimeException("failed to read binary quartets: " + inputPath, e);
        }

        long readElapsed = (System.currentTimeMillis() - start) / 1000;
        System.out.println("number of quartet = " + records);
        System.out.println("number of Taxa = " + taxMap.size());
        System.out.println("Quartet Reading Time : " + readElapsed + " seconds");
        return solveFromQuartetList(qr, taxMap);
    }

    private static String runTextInput(String inputPath, String inputFormat) {
        if ("newick".equals(inputFormat)) {
            return Routines.newickQuartetWeightAsFrequency(inputPath);
        }
        return Routines.readQuartetQMC(inputPath);
    }

    private static void writeOutputTree(String outputPath, String treeText) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath))) {
            bw.write(treeText);
        }
    }

    private static int runFastCli(String[] args) {
        String inputPath = null;
        String outputPath = null;
        String inputFormat = "bin";
        String taxonPrefix = DEFAULT_TAXON_PREFIX;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--input".equals(arg) && i + 1 < args.length) {
                inputPath = args[++i];
            } else if ("--output".equals(arg) && i + 1 < args.length) {
                outputPath = args[++i];
            } else if ("--input-format".equals(arg) && i + 1 < args.length) {
                inputFormat = args[++i];
            } else if ("--taxon-prefix".equals(arg) && i + 1 < args.length) {
                taxonPrefix = args[++i];
            } else {
                System.err.println("Unknown or incomplete argument: " + arg);
                return 2;
            }
        }

        if (inputPath == null || outputPath == null) {
            System.err.println("Usage: java -jar QFM-FI-fast.jar --input <file> --output <file> [--input-format bin|qmc|newick] [--taxon-prefix t]");
            return 2;
        }

        long start = System.currentTimeMillis();
        try {
            String treeText;
            if ("bin".equals(inputFormat)) {
                treeText = runBinaryInput(inputPath, taxonPrefix);
            } else if ("qmc".equals(inputFormat) || "newick".equals(inputFormat)) {
                treeText = runTextInput(inputPath, inputFormat);
            } else {
                System.err.println("Unsupported input format: " + inputFormat);
                return 2;
            }
            writeOutputTree(outputPath, treeText);
            long elapsed = (System.currentTimeMillis() - start) / 1000;
            System.out.println("Total Running Time : " + elapsed + " seconds");
            return 0;
        } catch (Exception e) {
            e.printStackTrace();
            return 1;
        }
    }

    public static void main(String[] args) {
        // Backward compatibility:
        // - Old mode: java -jar QFM-FI-fast.jar input output [quartetType]
        // - Fast mode: java -jar QFM-FI-fast.jar --input ... --output ... --input-format ...
        if (args.length >= 2 && !args[0].startsWith("--")) {
            Main.main(args);
            return;
        }
        int code = runFastCli(args);
        if (code != 0) {
            System.exit(code);
        }
    }
}
