
        package runners;
import org.junit.runner.RunWith;
import runners.Cucumber;
import runners.CucumberOptions;


@RunWith(Cucumber.class)
@CucumberOptions(
        plugin = {
                "pretty",
                "html:target/default-cucumber-reports.html"
        },
        monochrome = true,
        features = "src/test/resources/features",
        glue = {
                "stepdefinitions",
                "hooks"
        },
        tags = "@deneme",
        dryRun = false
)
class CukesRunner {
}