package hooks;

import utilities.ConfigReader;
import utilities.Driver;

import java.awt.*;

public class Hooks {




        @Before
        public void setUp(){

            String browser = ConfigReader.get("browser");
            if(browser.equalsIgnoreCase("chrome-headless")){
                Driver.get().manage().window().setSize(new Dimension(1920,1080));
            }else {
                Driver.get().manage().window().maximize();
            }

//        Driver.get().manage().timeouts().implicitlyWait(Duration.ofSeconds(10));

        }
        @After
        public void tearDownScenarios(Scenario scenario) {
            //System.out.println("After Metotu");
            if (scenario.isFailed()) {
                byte[] failedScreenshot = (byte[]) ((TakesScreenshot) Driver.get()).getScreenshotAs(OutputType.BYTES);
                scenario.attach(failedScreenshot, "image/png", "failed_scenario_" + scenario.getName());
                Driver.closeDriver();
            }

        }
    }

}
