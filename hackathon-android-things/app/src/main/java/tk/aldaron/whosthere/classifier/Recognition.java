package tk.aldaron.whosthere.classifier;

public class Recognition {
    private final String id;
    private final String name;
    private final Float confidence;

    public Recognition(final String id, final String name, final Float confidence){
        this.id = id;
        this.name = name;
        this.confidence = confidence;
    }

    public String getId(){
        return id;
    }

    public String getName(){
        return name;
    }

    public Float getConfidence(){
        return confidence == null ? 0f : confidence;
    }

    @Override
    public String toString(){
        String result = "";

        if (name!=null){
            result += name;
        }else{
            result += "unknown person";
        }

        return result.trim();
    }
}
