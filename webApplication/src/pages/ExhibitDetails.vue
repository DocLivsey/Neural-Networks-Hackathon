<script>
import {defineComponent} from 'vue'
import ExhibitCharacteristic from "@/components/ExhibitCharacteristic/ExhibitCharacteristic.vue";
import ExhibitionImages from "@/components/ExhibitionImages/ExhibitionImages.vue";

export default defineComponent({
    name: "ExhibitDetails",
    components: {ExhibitionImages, ExhibitCharacteristic}
})
</script>

<template>
    <div class="details">
        <ExhibitCharacteristic characteristic="Название" characteristic-value="Рождение Давида, Микелянджело"/>
        <ExhibitionImages class="margin-top-20"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Период" characteristic-value="Эпоха Возрождения"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Материал, техника" characteristic-value="Мрамор"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Размер" characteristic-value="200x100, 100x230"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Местонахождение" characteristic-value="Venice Museum"/>
        <div class="divider margin-top-25"></div>
        <ExhibitCharacteristic class="margin-top-15" characteristic="Описание" characteristic-value="Эпоха Возрождения"/>
        <div class="divider margin-top-25"></div>
        <ExhibitCharacteristic class="margin-top-15" characteristic="Номер по КП (ГИК)" characteristic-value="50272907"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Номер в Госкаталоге" characteristic-value="ПКГ КП-13258"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Инвентарный номер" characteristic-value="7447-гр"/>
        <ExhibitCharacteristic class="margin-top-20" characteristic="Внешняя ссылка" characteristic-value="-"/>
    </div>
</template>

<style scoped>
.divider {
    width: 17.8125em;
    height: 0.25em;
    border-radius: 1em;
    background-color: var(--fifth-color);
}
.details {
    display: flex;
    flex-direction: column;
    align-items: start;
    max-width: 17.875em;
    margin: 1.625em auto 2.5em;
}
.margin-top-15 {
    margin-top: 0.9375em;
}
.margin-top-20 {
    margin-top: 1.25em;
}
.margin-top-25 {
    margin-top: 1.5625em;
}
</style>