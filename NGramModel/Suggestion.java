package org.NGramModel;

public class Suggestion {
    private Object obj;
    private double score;
    private String type;

    public Suggestion(Object obj, double score, String type) {
        this.obj = obj;
        this.score = score;
        this.type = type;
    }

    public Object getSuggestion() {
        return obj;
    }

    public double getScore() {
        return score;
    }

    public String getType() {
        return type;
    }

    public void updateScore(double score) {
        this.score = score;
    }
}
