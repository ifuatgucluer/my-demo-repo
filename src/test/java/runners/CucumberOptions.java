package runners;

public @interface CucumberOptions {
    String[] plugin();

    boolean monochrome();

    String features();

    String tags();

    String[] glue();

    boolean dryRun();
}
